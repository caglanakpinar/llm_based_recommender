# User Generated Prompt

Start with User ID and describe the user profile using placeholders.

- User ID is {user_id}.
- Segment of {user_id} is {segment}.
- Notes for {user_id}: {notes}.
- Generated prompt for {user_id}: {generated_prompt}.
- Generated prompt embedding vectors for {user_id}: {generated_prompt_emb_vectors}.

## Single-Line Prompt Template
user_id: {user_id}; segment: {segment}; notes: {notes}; generated_prompt: {generated_prompt}; generated_prompt_emb_vectors: {generated_prompt_emb_vectors}

## Key Features
- Profile features:
	- segment of {user_id} is {segment}
	- profile completeness score of {user_id} is {profile_completeness_score}
	- profile age in days of {user_id} is {profile_age_days}

- Activity features:
	- number of total interactions by {user_id} is {num_total_interactions}
	- number of active days for {user_id} in last {lookback_window_days} days is {num_active_days}
	- recency of last interaction for {user_id} in days is {days_since_last_interaction}

- Preference features:
	- top categories for {user_id} are {top_categories}
	- category concentration score for {user_id} is {category_concentration_score}
	- top tags for {user_id} are {top_tags}
	- tag diversity score for {user_id} is {tag_diversity_score}

- Intent features:
	- purchase count of {user_id} is {num_purchases_user}
	- like count of {user_id} is {num_likes_user}
	- click-through rate of {user_id} is {ctr_user}
	- purchase conversion rate of {user_id} is {purchase_conversion_rate_user}

- Price behavior features:
	- average viewed item price for {user_id} is {avg_view_price_user}
	- average purchased item price for {user_id} is {avg_purchase_price_user}
	- price sensitivity score for {user_id} is {price_sensitivity_score}

- Collaborative features:
	- number of similar users to {user_id} is {num_similar_users}
	- average similarity to nearest users for {user_id} is {avg_similarity_to_neighbors}
	- peer influence score for {user_id} is {peer_influence_score}

- Risk and lifecycle features:
	- churn risk score for {user_id} is {churn_risk_score}
	- reactivation likelihood for {user_id} is {reactivation_likelihood}
	- lifecycle stage for {user_id} is {lifecycle_stage}

- Model-ready features:
	- user embedding vector for {user_id} is {user_embedding_vector}
	- generated prompt embedding vector for {user_id} is {generated_prompt_emb_vectors}
	- final user affinity prior score for {user_id} is {user_affinity_prior_score}
