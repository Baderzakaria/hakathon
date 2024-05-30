import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import streamlit as st
import matplotlib.pyplot as plt

# Load Data
@st.cache_data
def load_data():
    df_predi = pd.read_csv('newdata/datasetreco.csv', delimiter=';')
    return df_predi

# Preprocess Data
def preprocess_data(df):
    agg_df = df.groupby('product_id').agg(
        p_views_sum=pd.NamedAgg(column='p_views', aggfunc='sum'),
        p_carts_sum=pd.NamedAgg(column='p_carts', aggfunc='sum'),
        p_purchases_sum=pd.NamedAgg(column='p_purchases', aggfunc='sum')
    ).reset_index()

    agg_df['rating'] = agg_df['p_views_sum'] * 2 + agg_df['p_carts_sum'] * 4 + agg_df['p_purchases_sum'] * 10
    df = df.merge(agg_df[['product_id', 'rating']], on='product_id', how='left')
    return df

# Create PCA and Correlation Matrix
def create_pca_and_correlation_matrix(df):
    df_purchased = df[df["is_purchase"] >= 1][['user_id', 'product_id', 'category', 'subcategory', 'subsubcategory', 'rating']].drop_duplicates()
    pivot_table = df_purchased.pivot(index='product_id', columns='user_id', values='rating').fillna(0)

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(pivot_table)
    
    pca = PCA(n_components=800)
    pca_result = pca.fit_transform(normalized_data)
    
    pca_df = pd.DataFrame(pca_result, index=pivot_table.index, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    product_correlation_matrix = pca_df.T.corr()
    
    return df_purchased, product_correlation_matrix

# Get Recommendations for a Selected User
def get_recommendations(df_purchased, product_correlation_matrix, selected_user):
    user_purchase_history = df_purchased[df_purchased['user_id'] == selected_user]
    user_recommendations = []
    for product in user_purchase_history['product_id']:
        if product in product_correlation_matrix.index:
            product_correlations = product_correlation_matrix[product].drop(product)
            user_recommendations.extend(product_correlations.index.tolist())

    unique_recommendations = list(set(user_recommendations) - set(user_purchase_history['product_id']))

    final_recommendations = []
    final_weights = []
    for product in unique_recommendations:
        if product in product_correlation_matrix.index:
            total_weight = sum(product_correlation_matrix.loc[product, user_purchase_history['product_id']])
            final_recommendations.append(product)
            final_weights.append(total_weight)

    sorted_recommendations = sorted(zip(final_recommendations, final_weights), key=lambda x: x[1], reverse=True)

    recommendations = {
        'products': [x[0] for x in sorted_recommendations[:5]],
        'weights': [x[1] for x in sorted_recommendations[:5]]
    }

    return recommendations


def main():
    st.title('Product Recommendation System')

    df_predi = load_data()
    df = preprocess_data(df_predi)
    df_purchased, product_correlation_matrix = create_pca_and_correlation_matrix(df)

    unique_users = df_purchased['user_id'].unique()
    selected_user = st.selectbox('Select a user', unique_users)

    if st.button('Get Recommendations'):
        user_purchase_history = df_purchased[df_purchased['user_id'] == selected_user]
        
        st.write(f"Purchase history of User {selected_user}:")
        st.dataframe(user_purchase_history[['product_id', 'category', 'subcategory', 'subsubcategory']].drop_duplicates())

        recommendations = get_recommendations(df_purchased, product_correlation_matrix, selected_user)
        recommended_product_details = df_predi[df_predi['product_id'].isin(recommendations['products'])]

        st.write("Recommended products and their details:")
        st.dataframe(recommended_product_details[['product_id', 'category', 'subcategory', 'subsubcategory']].drop_duplicates())

if __name__ == '__main__':
    main()
