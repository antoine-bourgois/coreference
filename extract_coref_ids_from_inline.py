import re
from collections import Counter
from difflib import SequenceMatcher
from collections import defaultdict


def get_gold_mapping(gold_tokens, gen_tokens):
    gold_low = [t.lower() for t in gold_tokens]
    gen_low = [t.lower() for t in gen_tokens]
    mapping = {}
    matched_g = set()
    matched_p = set()

    g_counts = Counter(gold_low)
    p_counts = Counter(gen_low)
    global_uniques = {w for w in g_counts if g_counts[w] == 1 and p_counts[w] == 1}

    seeds = []
    for g_idx, word in enumerate(gold_low):
        if word in global_uniques:
            p_idx = gen_low.index(word)
            seeds.append((g_idx, p_idx))
    seeds.sort()

    for g_idx, p_idx in seeds:
        if g_idx in matched_g or p_idx in matched_p: continue
        mapping[p_idx] = g_idx
        matched_g.add(g_idx);
        matched_p.add(p_idx)

        lg, lp = g_idx - 1, p_idx - 1
        while lg >= 0 and lp >= 0:
            if lg in matched_g or lp in matched_p: break
            if gold_low[lg] == gen_low[lp]:
                mapping[lp] = lg
                matched_g.add(lg);
                matched_p.add(lp)
                lg -= 1;
                lp -= 1
            else:
                break

        rg, rp = g_idx + 1, p_idx + 1
        while rg < len(gold_low) and rp < len(gen_low):
            if rg in matched_g or rp in matched_p: break
            if gold_low[rg] == gen_low[rp]:
                mapping[rp] = rg
                matched_g.add(rg);
                matched_p.add(rp)
                rg += 1;
                rp += 1
            else:
                break

    def solve_gap_in_place(g_start, g_end, p_start, p_end):
        if g_start > g_end or p_start > p_end: return
        g_slice = gold_low[g_start:g_end + 1]
        p_slice = gen_low[p_start:p_end + 1]
        g_c, p_c = Counter(g_slice), Counter(p_slice)
        local_uniques = {w for w in g_c if g_c[w] == 1 and p_c[w] == 1}

        local_anchors = []
        if local_uniques:
            for i, word in enumerate(g_slice):
                if word in local_uniques:
                    j = p_slice.index(word)
                    gl_g, gl_p = g_start + i, p_start + j
                    if gl_p not in mapping:
                        mapping[gl_p] = gl_g
                        local_anchors.append((gl_g, gl_p))

        if local_anchors:
            local_anchors.sort()
            cg, cp = g_start, p_start
            for ng, np in local_anchors:
                solve_gap_in_place(cg, ng - 1, cp, np - 1)
                cg, cp = ng + 1, np + 1
            solve_gap_in_place(cg, g_end, cp, p_end)
        else:
            curr_p_fuzzy = p_start
            for curr_g in range(g_start, g_end + 1):
                target = gold_low[curr_g]
                best_s, best_p = 0.8, -1
                for p_search in range(curr_p_fuzzy, p_end + 1):
                    if p_search in mapping: continue
                    cand = gen_low[p_search]
                    score = 1.0 if target == cand else SequenceMatcher(None, target, cand).ratio()
                    if score >= best_s:
                        best_s, best_p = score, p_search
                        if score == 1.0: break
                if best_p != -1:
                    mapping[best_p] = curr_g
                    curr_p_fuzzy = best_p + 1

    sorted_p = sorted(mapping.keys())
    p_boundaries = [-1] + sorted_p + [len(gen_low)]
    for i in range(len(p_boundaries) - 1):
        p_s_idx, p_e_idx = p_boundaries[i], p_boundaries[i + 1]
        g_s_bound = mapping[p_s_idx] + 1 if p_s_idx != -1 else 0
        g_e_bound = mapping[p_e_idx] - 1 if p_e_idx != len(gen_low) else len(gold_low) - 1
        solve_gap_in_place(g_s_bound, g_e_bound, p_s_idx + 1, p_e_idx - 1)

    return mapping

def tokenize_bracket_text(text, TOKEN_RE = re.compile(r"\{___|\{\d+|\}|[^\s{}]+")):
    """
    Splits text into word tokens + bracket events.
    Returns:
      words: list[str]              -- plain word tokens, in order
      events: list[(kind, val, wpos)]  kind in {open_noid, open_id, close}
              wpos = number of words appended so far at the moment
              the bracket char was seen (i.e. index of the *next* word)
    """
    words = []
    events = []
    for chunk in TOKEN_RE.findall(text):
        if chunk == "{___":
            events.append(("open_noid", None, len(words)))
        elif re.fullmatch(r"\{\d+", chunk):
            events.append(("open_id", chunk[1:], len(words)))
        elif chunk == "}":
            events.append(("close", None, len(words)))
        else:
            words.append(chunk)
    return words, events

def build_spans(words, events):
    """
    Stack-parses the bracket events into spans, preserving NESTING via LIFO close.
    Returns a list of span dicts {'start','end','id'} IN THE ORDER THE BRACKETS
    WERE OPENED (i.e. reading order) -- since each dict is mutated in place when
    its closing '}' is found, end/id are correct by the time we return.
    """
    stack = []
    open_order = []
    for kind, val, wpos in events:
        if kind in ("open_noid", "open_id"):
            span = {"start": wpos, "end": None, "id": val}
            stack.append(span)
            open_order.append(span)
        elif kind == "close":
            if stack:
                span = stack.pop()
                span["end"] = wpos - 1  # last word actually included
    # auto-close anything left dangling (shouldn't normally happen)
    while stack:
        span = stack.pop()
        span["end"] = len(words) - 1
    return open_order

def fill_none_with_new_ids(aligned_coref_ids):
    """
    Replaces every None in aligned_coref_ids with a fresh unique id
    (as a string, to match the existing id format), strictly greater
    than any id already present. Nones are filled in list order, so
    the first None gets max+1, the second None gets max+2, etc.
    """
    used_ints = [int(i) for i in aligned_coref_ids if i is not None]
    next_id = (max(used_ints) + 1) if used_ints else 0

    filled = []
    for i in aligned_coref_ids:
        if i is None:
            filled.append(str(next_id))
            next_id += 1
        else:
            filled.append(i)

    mapping = {}
    sorted_ids = []
    for old_id in filled:
        if old_id not in mapping:
            mapping[old_id] = str(len(mapping))
        sorted_ids.append(mapping[old_id])

    return sorted_ids

def realign_coref_ids(input_text, pred_text, verbose=False):
    """
    Returns a list of ids (str) / None, one per '{___' span in input_text,
    in the order those spans open in input_text.
    """
    words_in, events_in = tokenize_bracket_text(input_text)
    words_pred, events_pred = tokenize_bracket_text(pred_text)

    spans_in = build_spans(words_in, events_in)  # gold spans, id=None
    spans_pred = build_spans(words_pred, events_pred)  # pred spans, id=actual

    # word-level alignment: pred word idx -> input word idx
    mapping = get_gold_mapping(words_in, words_pred)

    if verbose:
        print(f"input words: {len(words_in)}  pred words: {len(words_pred)}  mapped: {len(mapping)}")
        print(f"input spans: {len(spans_in)}  pred spans: {len(spans_pred)}")

    # translate every pred span into gold(input) token-space
    translated = []
    for sp in spans_pred:
        g_start = mapping.get(sp["start"])
        g_end = mapping.get(sp["end"])
        if g_start is not None and g_end is not None:
            translated.append({"start": g_start, "end": g_end, "id": sp["id"], "used": False})

    results = []
    for gold_sp in spans_in:
        gs, ge = gold_sp["start"], gold_sp["end"]
        # pass 1: exact (start,end) match
        best = None
        for cand in translated:
            if cand["used"]:
                continue
            if cand["start"] == gs and cand["end"] == ge:
                best = cand
                break
        # pass 2: same start, closest end
        if best is None:
            same_start = [c for c in translated if not c["used"] and c["start"] == gs]
            if same_start:
                best = min(same_start, key=lambda c: abs(c["end"] - ge))
        if best is not None:
            best["used"] = True
            results.append(best["id"])
        else:
            results.append(None)

    results = fill_none_with_new_ids(results)

    return results, {
        "words_in": words_in, "words_pred": words_pred,
        "spans_in": spans_in, "spans_pred": spans_pred,
        "mapping": mapping, "translated": translated,
    }

def extract_predicted_COREF_ids_from_output(input, text, remove_singletons=False):
    element = {"text": text}
    element["opening_count"] = element["text"].count("{")
    element["closing_count"] = element["text"].count("}")

    aligned_coref_ids, _ = realign_coref_ids(input, text, verbose=False)

    element["coref_ids"] = aligned_coref_ids

    clusters = defaultdict(list)

    for i, COREF_id in enumerate(element["coref_ids"]):
        clusters[COREF_id].append([i])

    element["clusters"] = list(dict(clusters).values())
    if remove_singletons:
        element["clusters"] = [cluster for cluster in element["clusters"] if len(cluster) > 1]

    return element["coref_ids"]