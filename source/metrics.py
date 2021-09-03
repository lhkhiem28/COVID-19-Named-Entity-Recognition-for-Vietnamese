
from seqeval.scheme import IOB2
from seqeval import metrics

def entity_f1_score(
    annotations, predictions
    , criterion_ignored_class, tag_names
    , average
):
    active_indices = annotations != criterion_ignored_class
    annotations, predictions = annotations[active_indices].tolist(), predictions[active_indices].tolist()
    annotations, predictions = [tag_names[cls] for cls in annotations], [tag_names[cls] for cls in predictions]

    f1 = metrics.f1_score(
        [annotations], [predictions]
        , mode="strict", scheme=IOB2
        , average=average
    )

    return f1

def classification_report(
    annotations, predictions
    , criterion_ignored_class, tag_names
):
    active_indices = annotations != criterion_ignored_class
    annotations, predictions = annotations[active_indices].tolist(), predictions[active_indices].tolist()
    annotations, predictions = [tag_names[cls] for cls in annotations], [tag_names[cls] for cls in predictions]

    report = metrics.classification_report(
        [annotations], [predictions]
        , mode="strict", scheme=IOB2
    )

    return report