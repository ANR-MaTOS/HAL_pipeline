from collections import defaultdict
import json

def clean(s: str) -> str:
    """Normalise whitespace."""
    return " ".join(s.split())

def align(data: dict) -> dict:
    # final_alignments = defaultdict(list)

    for docid, entry in data.items():
        if not entry.get("alignments"):
            continue
        
        alignments = entry["alignments"]
        entry["final_alignments"] = []
        # Build flat lists for every stream 

        def _flat(key):
            return (
                [item["src_sent"].strip() for item in alignments[key]],
                [item["tgt_sent"].strip() for item in alignments[key]],
            )

        doc0_src, doc0_tgt = _flat("doc_0shot")
        doc1_src, doc1_tgt = _flat("doc_1shot")
        doc2_src, doc2_tgt = _flat("doc_2shot")

        # Stream descriptors (name, src_list, tgt_list) 
        streams = [
            ("doc0",  doc0_src,  doc0_tgt),
            ("doc1",  doc1_src,  doc1_tgt),
            ("doc2",  doc2_src,  doc2_tgt),
        ]
        n = len(streams)

        # Per-stream state 
        indices  = [0] * n          # next sentence index for each stream
        src_bufs = [""] * n         # accumulated source text
        tgt_bufs = [""] * n         # accumulated target text

        def exhausted(si):
            return indices[si] >= len(streams[si][1])

        def load_next(si):
            """Append the next sentence of stream si into its buffer."""
            _, src_list, tgt_list = streams[si]
            idx = indices[si]
            if idx < len(src_list):
                sep = " " if src_bufs[si] else ""
                src_bufs[si] += sep + src_list[idx]
                tgt_bufs[si] += (" " if tgt_bufs[si] else "") + tgt_list[idx]
                indices[si] += 1
                return True
            return False

        # Pre-load one sentence into every stream
        for si in range(n):
            load_next(si)

        # Main alignment loop 
        while any(src_bufs[si] for si in range(n)):

            cleaned = [clean(src_bufs[si]) for si in range(n)]

            # All streams agree → emit and reset
            if len(set(cleaned)) == 1:
                # final_alignments[docid].append({
                entry["final_alignments"].append({
                    "source": cleaned[0],
                    "translations": {
                        "doc0_level":    tgt_bufs[0],
                        "doc1_level":    tgt_bufs[1],
                        "doc2_level":    tgt_bufs[2],
                    },
                })
                src_bufs = [""] * n
                tgt_bufs = [""] * n
                # Load next sentence for every stream
                for si in range(n):
                    if not exhausted(si):
                        load_next(si)
                continue

            # Not aligned yet: expand the shortest non-empty buffer(s)
            lengths = [(len(cleaned[si]), si) for si in range(n) if cleaned[si]]
            min_len = lengths[0][0]
            for l, _ in lengths:
                if l < min_len:
                    min_len = l

            expanded = False
            for _, si in lengths:
                if len(cleaned[si]) == min_len:
                    if not exhausted(si):
                        load_next(si)
                        expanded = True
                    # else this stream is stuck at its last sentence —
                    # keep it as-is and let others catch up

            if not expanded:
                # Every stream at min length is already exhausted.
                # Emit what we have (best-effort) and stop.
                print(f"[{docid}] Alignment dead-end — emitting remainder:")
                for si in range(n):
                    print(f"  {streams[si][0]}: {cleaned[si]!r}")
                
                # final_alignments[docid].append({
                entry["final_alignments"].append({
                    "source": cleaned[0] or next(
                        (cleaned[si] for si in range(n) if cleaned[si]), ""
                    ),
                    "translations": {
                        "doc0_level":    tgt_bufs[0],
                        "doc1_level":    tgt_bufs[1],
                        "doc2_level":    tgt_bufs[2],
                    },
                })
                break

    # return dict(final_alignments)
    return data