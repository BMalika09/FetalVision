from fetalvision.evaluation import compute_metrics


def test_perfect_predictions():
    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 2, 3, 4, 5]

    results = compute_metrics(y_true, y_pred)

    assert results["overall"]["accuracy"] == 1.0
    assert results["overall"]["macro_f1"] == 1.0
    assert len(results["per_class"]) == 6


def test_missing_predicted_class():
    """Check metrics when a class is never predicted."""

    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 1, 3, 4, 5]

    results = compute_metrics(y_true, y_pred)

    femur_metrics = results["per_class"]["Fetal femur"]

    assert femur_metrics["precision"] == 0.0
    assert femur_metrics["recall"] == 0.0
    assert femur_metrics["f1"] == 0.0
    assert femur_metrics["support"] == 1

    assert results["overall"]["accuracy"] == 5 / 6
    assert results["overall"]["macro_f1"] < 1.0