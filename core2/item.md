# Item Generated Prompt

Start with Item ID and describe the item using placeholders.

- Item ID is **{item_id}**.
- Title of {item_id} is {title}.
- Category of {title} is {category}.
- Tags of {title} are {tags}.
- Price of {title} is {price}.
- Description of {title}: {description}.

## Single-Line Prompt Template
item_id: {item_id}; title: {title}; category: {category}; tags: {tags}; price: {price}; description: {description}

## Key Features
- Popularity features:
	- **number of unique users** that interacted with **{item_id}** is **{num_unique_users_interacted}**
	- **number of purchases** for **{item_id}** is **{num_purchases_item}
	- **purchase rate of {item_id}** is **{purchase_rate_item}**

- **Collaborative filtering features**:
	- **number of unique users** that both interacted with **{item_id}** and items seen by target_user is **{num_common_users_item_target}**
	- **number of similar users** to target who purchased **{item_id}** is **{num_similar_users_purchased_item}**
	- **average similarity score of users who engaged** with **{item_id}** is **{avg_similarity_users_item}**

- **Content affinity features**:
	- category match between target_user preference and **{item_id}** is **{category_match_score}**
	- **tag overlap score between target_user interests** and **{item_id} tags** is **{tag_overlap_score}**
	- **price fit score of {item_id}** for target_user is **{price_fit_score}**

- **Temporal features**:
	- **days since last interaction on {item_id}** is **{days_since_last_item_interaction}**
	- **interaction trend for {item_id}** in last **{lookback_window_days}** days is **{recent_trend_score}**
	- **recency weight of {item_id}** for target_user is **{item_recency_weight}**

- **Risk and exclusion features**:
	- **target_user already purchased {item_id}: {already_purchased_flag}**
	- **target_user disliked or removed {item_id}: {negative_feedback_flag}**
	- **final eligibility of {item_id} for {user_id} is {eligible_flag}**
