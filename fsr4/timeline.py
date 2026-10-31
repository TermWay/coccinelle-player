"""Which frames of a video are upscaled, and where the renderer should work next.

The renderer writes the upscaled video as pieces: consecutive frames encoded into one file each. A piece is
`pending` while its encoder finishes and `ready` once the file is complete (playable, joinable). Pieces never
overlap, and the ready pieces from frame 0 to the end join into the final file. The frame count is the probe's
estimate until a read reaches the end of the video (`exact`); until then the frames past it count as missing,
so the renderer reads on until the decoder runs out.

Scheduling follows the playhead. In preview mode the renderer keeps the frames just ahead of the playhead
upscaled (it starts when less than `low_water` frames are ready ahead and stops at `high_water`); in batch
mode it upscales everything, starting at the playhead and then filling what is left from the beginning.
"""
from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(eq=False)
class Piece:
    """Frames [start, end) encoded into `path`."""
    start: int
    end: int
    path: Path
    ready: bool = False

    @property
    def frames(self) -> int:
        return self.end - self.start


class Timeline:
    """The pieces of one upscale, sorted and disjoint. Every method is thread-safe.

    `busy` arguments name the frames a run has written but not yet handed over as a piece (start, end).
    """

    def __init__(self, total: int, exact: bool = False):
        self._lock = threading.Lock()
        self._pieces: list[Piece] = []
        self._starts: list[int] = []
        self.total = max(1, total)
        self.exact = exact

    # ---- changes --------------------------------------------------------------------------------

    def add(self, piece: Piece) -> None:
        """Add a pending piece; extends `total` when the piece runs past an estimated end."""
        if piece.end <= piece.start:
            raise ValueError(f"empty piece {piece.start}-{piece.end}")
        with self._lock:
            index = bisect.bisect_right(self._starts, piece.start)
            if index > 0 and self._pieces[index - 1].end > piece.start:
                raise ValueError(f"piece {piece.start}-{piece.end} overlaps {self._pieces[index - 1].start}-"
                                 f"{self._pieces[index - 1].end}")
            if index < len(self._pieces) and self._pieces[index].start < piece.end:
                raise ValueError(f"piece {piece.start}-{piece.end} overlaps {self._pieces[index].start}-"
                                 f"{self._pieces[index].end}")
            self._pieces.insert(index, piece)
            self._starts.insert(index, piece.start)
            self.total = max(self.total, piece.end)

    def mark_ready(self, piece: Piece) -> None:
        with self._lock:
            piece.ready = True

    def remove(self, piece: Piece) -> None:
        with self._lock:
            index = self._pieces.index(piece)
            del self._pieces[index]
            del self._starts[index]

    def clear(self) -> list[Piece]:
        """Forget every piece (settings changed); returns them so their files can be deleted."""
        with self._lock:
            removed, self._pieces, self._starts = self._pieces, [], []
            return removed

    def set_total(self, total: int) -> None:
        """The real frame count, found when a read reached the end of the file."""
        with self._lock:
            last_end = self._pieces[-1].end if self._pieces else 0
            self.total = max(1, total, last_end)
            self.exact = True

    # ---- queries --------------------------------------------------------------------------------

    def pieces(self, ready_only: bool = False) -> list[Piece]:
        with self._lock:
            return [piece for piece in self._pieces if piece.ready or not ready_only]

    def piece_at(self, frame: int, ready_only: bool = False) -> Piece | None:
        with self._lock:
            index = bisect.bisect_right(self._starts, frame) - 1
            if index >= 0:
                piece = self._pieces[index]
                if piece.start <= frame < piece.end and (piece.ready or not ready_only):
                    return piece
            return None

    def _covered_from(self, frame: int, ready_only: bool, busy: tuple[int, int] | None) -> int:
        """First frame at or after `frame` that is not covered (pieces, plus `busy`)."""
        index = max(0, bisect.bisect_right(self._starts, frame) - 1)
        position = frame
        while True:
            if busy is not None and busy[0] <= position < busy[1]:
                position = busy[1]
                continue
            while index < len(self._pieces) and self._pieces[index].end <= position:
                index += 1
            if index < len(self._pieces):
                piece = self._pieces[index]
                if piece.start <= position and (piece.ready or not ready_only):
                    position = piece.end
                    continue
            return position

    def ready_until(self, frame: int) -> int:
        """First frame at or after `frame` that is not playable yet (`frame` itself when it is not)."""
        with self._lock:
            return self._covered_from(frame, True, None)

    def next_missing(self, frame: int, busy: tuple[int, int] | None = None) -> int | None:
        """First frame at or after `frame` that no piece (ready or pending) nor `busy` covers; None if none.
        Past an estimated end every frame is missing: only a read that reaches the end settles it."""
        with self._lock:
            position = self._covered_from(max(0, frame), False, busy)
            return position if position < self.total or not self.exact else None

    def gap_end(self, frame: int) -> int:
        """End of the uncovered stretch that starts at `frame`: the next piece's start, or `total`."""
        with self._lock:
            index = bisect.bisect_right(self._starts, frame)
            return self._pieces[index].start if index < len(self._pieces) else self.total

    def next_ready_start(self, frame: int) -> int | None:
        """`frame` if a ready piece covers it, else the start of the next ready piece; None if there is none."""
        with self._lock:
            for piece in self._pieces:
                if piece.ready and piece.end > frame:
                    return max(frame, piece.start)
            return None

    def ready_frames(self) -> int:
        with self._lock:
            return sum(piece.frames for piece in self._pieces if piece.ready)

    def complete(self) -> bool:
        with self._lock:
            return self.exact and self._covered_from(0, True, None) >= self.total

    def ready_ranges(self) -> list[tuple[int, int]]:
        """The playable frames as merged (start, end) ranges, for drawing."""
        with self._lock:
            ranges: list[tuple[int, int]] = []
            for piece in self._pieces:
                if not piece.ready:
                    continue
                if ranges and ranges[-1][1] == piece.start:
                    ranges[-1] = (ranges[-1][0], piece.end)
                else:
                    ranges.append((piece.start, piece.end))
            return ranges

    def ready_prefix(self) -> list[Piece]:
        """The ready pieces that join into frames 0..n without a gap."""
        with self._lock:
            prefix = []
            position = 0
            for piece in self._pieces:
                if piece.start != position or not piece.ready:
                    break
                prefix.append(piece)
                position = piece.end
            return prefix


def next_start(timeline: Timeline, target: int, batch: bool, low_water: int) -> int | None:
    """Where an idle renderer should start, or None when nothing is needed now."""
    missing = timeline.next_missing(target)
    if batch:
        return missing if missing is not None else timeline.next_missing(0)
    if missing is None or missing - target >= low_water:
        return None
    return missing


def keep_going(timeline: Timeline, target: int, batch: bool, high_water: int, busy: tuple[int, int] | None,
               position: int) -> bool:
    """Whether a run that is about to render frame `position` should go on (else it stops after a seek, a
    finished stretch, or once far enough ahead of the playhead in preview mode)."""
    needed = timeline.next_missing(target, busy)
    if needed is None and batch:
        needed = timeline.next_missing(0, busy)
    if needed != position:
        return False
    return batch or position - target < high_water


def piece_lengths(frame_rate: float) -> Iterator[int]:
    """Frames per piece within a run: short at first so playback can start soon after a seek, then longer
    (fewer files and keyframes): 0.5 s, 1 s, 2 s, 4 s, then 8 s."""
    for seconds in (0.5, 1, 2, 4):
        yield max(1, int(round(frame_rate * seconds)))
    while True:
        yield max(1, int(round(frame_rate * 8)))
