
from seqeval.scheme import IOB2
from seqeval import metrics

def entity_f1_score(
    anno, pred
    , criterion_ignored_la, tag_names
    , average
):
    active_indices = anno != criterion_ignored_la
    anno, pred = anno[active_indices].tolist(), pred[active_indices].tolist()
    anno, pred = [tag_names[tag] for tag in anno], [tag_names[tag] for tag in pred]

    f1_score = metrics.f1_score(
        [anno], [pred]
        , mode="strict", scheme=IOB2
        , average=average
    )

    return f1_score

def entity_classification_report(
    anno, pred
    , criterion_ignored_la, tag_names
):
    active_indices = anno != criterion_ignored_la
    anno, pred = anno[active_indices].tolist(), pred[active_indices].tolist()
    anno, pred = [tag_names[tag] for tag in anno], [tag_names[tag] for tag in pred]

    classification_report = metrics.classification_report(
        [anno], [pred]
        , mode="strict", scheme=IOB2
        , digits=3
    )

    return classification_report