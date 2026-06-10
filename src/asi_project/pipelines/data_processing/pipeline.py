"""Definicja potoku 'data_processing'."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data, encode_features, engineer_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs="hotels_raw",
                outputs="hotels_clean",
                name="clean_data_node",
            ),
            node(
                func=engineer_features,
                inputs=["hotels_clean", "params:top_n_countries"],
                outputs="hotels_features",
                name="engineer_features_node",
            ),
            node(
                func=encode_features,
                inputs=["hotels_features", "params:target_column"],
                outputs=["model_input", "model_features"],
                name="encode_features_node",
            ),
        ]
    )
