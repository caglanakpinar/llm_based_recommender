


from core.configs import Configs


def generate_recommender_engine( 
        engine_name: str,
        item_data_path: str,
        user_data_path: str,
        user_item_data_path: str,
        top_k: int = 10,
        constraints: str | None = None,
        user_profile_columns: dict[str, str] | None = None,
        item_catalog_columns: dict[str, str] | None = None
    ):
    """Generate a recommender engine based on the provided parameters. """  
    # Your implementation here
    Configs.create(
        engine_name=engine_name,
        item_data_path=item_data_path,
        user_data_path=user_data_path,
        user_item_data_path=user_item_data_path,
        top_k=top_k,
        constraints=constraints,
        user_profile_columns=user_profile_columns,
        item_catalog_columns=item_catalog_columns
    )
