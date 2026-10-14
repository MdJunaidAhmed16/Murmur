"""What the user actually heard: audio sent per sentence, mapped back to words."""

from dataclasses import dataclass, field


@dataclass
class _Segment:
    text: str
    start: int  # session sample offset where this sentence's audio begins
    end: int


@dataclass
class PlaybackLedger:
    """Positions are in samples since the session started, the same count the browser keeps.

    Sent is not heard: audio goes out faster than real time and sits in the browser's queue.
    When playback is cut, the browser reports how far it got and heard_text() maps that
    position back to words. Without a report, the caller estimates from wall-clock time.
    """

    sample_rate: int
    session_offset: int  # samples sent in this session before this turn
    segments: list[_Segment] = field(default_factory=list)

    @property
    def sent_end(self) -> int:
        return self.segments[-1].end if self.segments else self.session_offset

    @property
    def sent_samples(self) -> int:
        return self.sent_end - self.session_offset

    def begin(self, text: str) -> None:
        self.segments.append(_Segment(text, self.sent_end, self.sent_end))

    def add_audio(self, samples: int) -> None:
        self.segments[-1].end += samples

    def heard_text(self, played_until: int) -> str:
        """Whole sentences played, plus the played fraction of the one cut off, by word count."""
        words: list[str] = []
        for seg in self.segments:
            if seg.end == seg.start:
                break  # synthesis begun, no audio yet: nothing of it can have been heard
            if played_until >= seg.end:
                words.append(seg.text)
            elif played_until > seg.start and seg.end > seg.start:
                fraction = (played_until - seg.start) / (seg.end - seg.start)
                tokens = seg.text.split()
                # Rounds down: claiming a word the user never heard is worse than dropping one.
                if cut := tokens[: int(len(tokens) * fraction)]:
                    words.append(" ".join(cut))
                break
            else:
                break
        return " ".join(words)

    def all_text(self) -> str:
        return " ".join(seg.text for seg in self.segments)
