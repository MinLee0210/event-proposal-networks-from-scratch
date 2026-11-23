"""Event representation and geometric helpers shared by losses, inference and metrics.

A sound event is `(class_id, onset, offset, score)` in seconds. The same
dataclass is used for ground truth (score ignored / set to 1.0), EPN proposals,
and final predictions. Keeping it tiny and dependency-free lets the
post-processing, inference and evaluation code agree on one representation.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Event:
    """A temporal sound event in seconds.

    Attributes:
        class_id: Integer class index.
        onset: Event start time in seconds.
        offset: Event end time in seconds.
        score: Confidence in `[0, 1]`; ground truth uses 1.0.
    """

    class_id: int
    onset: float
    offset: float
    score: float = 1.0

    @property
    def duration(self) -> float:
        """Return the non-negative event duration in seconds.

        Returns:
            `max(offset - onset, 0.0)`.
        """
        return max(self.offset - self.onset, 0.0)

    def clipped(self, start: float, end: float) -> "Event":
        """Clip the event to the interval `[start, end]`.

        Args:
            start: Lower time bound in seconds.
            end: Upper time bound in seconds.

        Returns:
            A new `Event` whose onset/offset lie inside the bounds.
        """
        return Event(
            self.class_id,
            min(max(self.onset, start), end),
            min(max(self.offset, start), end),
            self.score,
        )


def intersection(a: Event, b: Event) -> float:
    """Compute the temporal intersection length of two events.

    Args:
        a: First event.
        b: Second event.

    Returns:
        Overlap in seconds, or 0.0 when the events do not overlap.
    """
    return max(0.0, min(a.offset, b.offset) - max(a.onset, b.onset))


def union(a: Event, b: Event) -> float:
    """Compute the temporal union length of two events.

    Args:
        a: First event.
        b: Second event.

    Returns:
        Union length in seconds.
    """
    return a.duration + b.duration - intersection(a, b)


def iou(a: Event, b: Event) -> float:
    """Compute temporal intersection-over-union.

    Args:
        a: First event.
        b: Second event.

    Returns:
        IoU in `[0, 1]`; 0.0 when either event has zero duration.
    """
    inter = intersection(a, b)
    if inter <= 0.0:
        return 0.0
    denom = a.duration + b.duration - inter
    return inter / denom if denom > 0 else 0.0


def overlaps(a: Event, b: Event) -> bool:
    """Return whether two events share any positive-length interval.

    Args:
        a: First event.
        b: Second event.

    Returns:
        True when `intersection(a, b) > 0`.
    """
    return intersection(a, b) > 0.0


def merge_events(events: Iterable[Event]) -> List[Event]:
    """Merge overlapping or abutting events of the same class.

    URBAN-SED is generated so that a source class does not overlap itself, but
    real annotations can. Merging keeps the frame-level presence label
    well-defined (one region per class) while the original onsets/offsets are
    only used through those merged boundaries.

    Args:
        events: Iterable of possibly overlapping events.

    Returns:
        Events sorted by `(onset, class_id)`, with same-class overlaps merged.
    """
    by_class: dict[int, List[Event]] = {}
    for ev in events:
        if ev.offset <= ev.onset:
            continue
        by_class.setdefault(ev.class_id, []).append(ev)

    merged: List[Event] = []
    for class_id, group in by_class.items():
        group.sort(key=lambda e: e.onset)
        cur = group[0]
        for ev in group[1:]:
            if ev.onset <= cur.offset:
                cur = Event(
                    class_id,
                    cur.onset,
                    max(cur.offset, ev.offset),
                    max(cur.score, ev.score),
                )
            else:
                merged.append(cur)
                cur = ev
        merged.append(cur)
    merged.sort(key=lambda e: (e.onset, e.class_id))
    return merged


def events_to_frames(
    events: Sequence[Event],
    num_classes: int,
    num_frames: int,
    frame_rate: float,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """Rasterise events into per-class presence / onset / offset frame labels.

    A frame covers `[t / frame_rate, (t + 1) / frame_rate)`; an event is active
    on every frame it overlaps. Onsets/offsets are placed on the first active
    frame and the first inactive frame after the event (or the last frame when
    the event hits the right clip boundary).

    Args:
        events: Strong annotations to rasterise.
        num_classes: Number of classes `C`.
        num_frames: Number of frames `T`.
        frame_rate: Frames per second.

    Returns:
        A tuple of three `C x T` nested lists: presence, onset and offset.
    """
    pres = [[0.0] * num_frames for _ in range(num_classes)]
    onset = [[0.0] * num_frames for _ in range(num_classes)]
    offset = [[0.0] * num_frames for _ in range(num_classes)]

    for ev in merge_events(events):
        if not 0 <= ev.class_id < num_classes:
            continue
        start = int(round(ev.onset * frame_rate))
        end = int(round(ev.offset * frame_rate))
        start = min(max(start, 0), num_frames - 1)
        end = min(max(end, start + 1), num_frames)
        onset[ev.class_id][start] = 1.0
        offset[ev.class_id][min(end, num_frames - 1)] = 1.0
        for t in range(start, end):
            pres[ev.class_id][t] = 1.0
    return pres, onset, offset


def frames_to_events(
    presence: Sequence[Sequence[float]],
    frame_rate: float,
    threshold: float = 0.5,
    min_duration: float = 0.0,
) -> List[Event]:
    """Recover events from a `C x T` presence array.

    Used to build ground-truth events at evaluation time (from the frame
    labels) without keeping a second, ragged copy of the annotations around.

    Args:
        presence: Per-class presence values, `C x T`.
        frame_rate: Frames per second.
        threshold: A frame is active when `value >= threshold`.
        min_duration: Drop shorter events (seconds).

    Returns:
        Events sorted by class and time.
    """
    events: List[Event] = []
    for class_id, row in enumerate(presence):
        active = False
        start = 0
        for t, value in enumerate(row):
            on = value >= threshold
            if on and not active:
                active, start = True, t
            elif not on and active:
                active = False
                onset_s, offset_s = start / frame_rate, t / frame_rate
                if offset_s - onset_s >= min_duration:
                    events.append(Event(class_id, onset_s, offset_s, 1.0))
        if active:
            onset_s, offset_s = start / frame_rate, len(row) / frame_rate
            if offset_s - onset_s >= min_duration:
                events.append(Event(class_id, onset_s, offset_s, 1.0))
    return events
