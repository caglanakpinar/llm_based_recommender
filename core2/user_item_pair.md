# User-Item Pair Generated Prompt

Start with User ID and Item ID, then describe the pair using placeholders.

- User ID is **{user_id}**.
- Item ID is **{item_id}**.
- Title of {item_id} is {title}.

## Single-Line Prompt Template
user_id: **{user_id}**; item_id: **{item_id}**; title: {title}

## Key Features
- **Pair interaction features**:
	- **number of interactions between {user_id} and {item_id}** is **{num_user_item_interactions}**
	- **last action by {user_id} on {item_id}** is **{last_action_user_item}**
	- **days since last interaction between {user_id} and {item_id}** is **{days_since_last_user_item_interaction}**

- **Pair intent features**:
	- **view count of {user_id} on {item_id}** is **{num_views_user_item}**
	- **click count of {user_id} on {item_id}** is **{num_clicks_user_item}**
	- **purchase flag for {user_id}-{item_id}** is **{purchased_user_item_flag}**
	- **dislike/remove flag for {user_id}-{item_id}** is **{negative_user_item_flag}**

- **Pair strength features**:
	- **weighted interaction score for {user_id}-{item_id}** is **{weighted_user_item_signal}**
	- **recency-weighted interaction score for {user_id}-{item_id}** is **{recency_weighted_user_item_signal}**
	- **confidence score for pair {user_id}-{item_id}** is **{pair_confidence_score}**

- **Global item context for pair**:
	- **popularity percentile of {item_id}** is **{item_popularity_percentile}**
	- **conversion rate of {item_id}** is **{item_conversion_rate}**
	- **novelty score of {item_id} for {user_id}** is **{pair_novelty_score}**

- **Collaborative pair features**:
	- **number of similar users who interacted with {item_id}** is **{num_similar_users_interacted_item}**
	- **number of similar users who purchased {item_id}** is **{num_similar_users_purchased_item_per_user}**
	- **peer agreement score for {user_id}-{item_id}** is **{peer_agreement_score}**

- **Sequence and session features**:
	- **probability that {item_id} is next action for {user_id}** is **{next_item_probability}**
	- **session co-occurrence score for {user_id}-{item_id}** is **{session_cooccurrence_score}**
	- **transition score from last seen item to {item_id} for {user_id}** is **{item_transition_score}**

- **Exclusion and eligibility features**:
	- **eligibility flag for recommending {item_id} to {user_id}** is **{eligible_flag}**
	- **exclusion reason code for {user_id}-{item_id}** is **{exclusion_reason_code}**
