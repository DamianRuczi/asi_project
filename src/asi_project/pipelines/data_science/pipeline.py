"""Definicja potoku 'data_science'."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    compare_models,
    evaluate_model,
    split_data,
    train_best_model,
    tune_best_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=[
                    "model_input",
                    "params:target_column",
                    "params:test_size",
                    "params:random_state",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=compare_models,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:model_comparison",
                    "params:random_state",
                ],
                outputs="model_comparison",
                name="compare_models_node",
            ),
            node(
                func=tune_best_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "model_comparison",
                    "parameters",
                ],
                outputs="optuna_results",
                name="tune_best_model_node",
            ),
            node(
                func=train_best_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "optuna_results",
                    "params:random_state",
                ],
                outputs="model",
                name="train_best_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test", "optuna_results", "parameters"],
                outputs="metrics",
                name="evaluate_model_node",
            ),
        ]
    )
