import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title='MTN Customer Churn Prediction App', layout='wide')
model = joblib.load("Mtn_customer_churn_model.pkl")

tab1, tab2 = st.tabs(["Prediction App tab", "Customer data Visualisation"])

with tab1:
    st.title('MTN Customer Churn Prediction APP')

    st.subheader('This apps speaks about mtn customer churn, it also shows the customers details about each customers in the second tab')
    st.subheader('Enter Details to predict the risk of customer churning')
    
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input('Customer Age?', min_value=18, max_value=85, value=18)
        State = st.selectbox('Customer State', options=[
            'Kwara', 'Abuja (FCT)', 'Sokoto', 'Gombe', 'Oyo', 'Plateau',
            'Jigawa', 'Imo', 'Bauchi', 'Ondo', 'Kebbi', 'Adamawa', 'Yobe',
            'Anambra', 'Cross River', 'Kogi', 'Osun', 'Kano', 'Benue',
            'Rivers', 'Enugu', 'Borno', 'Edo', 'Kaduna', 'Abia', 'Ekiti',
            'Bayelsa', 'Delta', 'Zamfara', 'Akwa Ibom', 'Nasarawa', 'Taraba',
            'Niger', 'Katsina', 'Lagos'
        ])

        MTN_Device = st.selectbox('Customer MTN Device', options=['4G Router', 'Mobile SIM Card', '5G Broadband Router', 'Broadband MiFi'])

        Gender = st.selectbox('Customer Gender', options=['Male', 'Female'])

        Customer_review = st.selectbox('Customer Review', options=['Fair', 'Poor', 'Good', 'Excellent', 'Very Good'])
        tenure = st.number_input('Customer Tenure (Months)', min_value=0, max_value=40, value=0)



    with col2:
        Unit_Price = st.number_input('Cost of Data Plan', min_value=0, max_value=500000, value=0)
        Number_of_Times_Purchased = st.number_input('Number of Times Purchased', min_value=0, max_value=20, value=0)
        Total_Revenue = st.number_input('Total Monthly Revenue', min_value=1, max_value=350000, value=1)
        Data_usage = st.number_input('Custormers Data Usage in Gigabyte(gb)', min_value=1, max_value=350000, value=1)
        Subscription_Plan = st.selectbox(
            'Customer Subscription Plan',
            options=[
                '165GB Monthly Plan', '12.5GB Monthly Plan', '150GB FUP Monthly Unlimited',
                '1GB+1.5mins Daily Plan', '30GB Monthly Broadband Plan', '10GB+10mins Monthly Plan',
                '25GB Monthly Plan', '7GB Monthly Plan', '1.5TB Yearly Broadband Plan',
                '65GB Monthly Plan', '120GB Monthly Broadband Plan', '300GB FUP Monthly Unlimited',
                '60GB Monthly Broadband Plan', '500MB Daily Plan', '3.2GB 2-Day Plan',
                '20GB Monthly Plan', '2.5GB 2-Day Plan', '450GB 3-Month Broadband Plan',
                '200GB Monthly Broadband Plan', '1.5GB 2-Day Plan', '16.5GB+10mins Monthly Plan'
            ]
        )

    # Mappings
    Gender_mapping = {'Male': 1, 'Female': 0}
    state_mapping = {
        'Kwara': 1, 'Abuja (FCT)': 2, 'Sokoto': 3, 'Gombe': 4, 'Oyo': 5, 'Plateau': 6,
        'Jigawa': 7, 'Imo': 8, 'Bauchi': 9, 'Ondo': 10, 'Kebbi': 11, 'Adamawa': 12, 'Yobe': 13,
        'Anambra': 14, 'Cross River': 15, 'Kogi': 16, 'Osun': 17, 'Kano': 18, 'Benue': 19,
        'Rivers': 20, 'Enugu': 21, 'Borno': 22, 'Edo': 23, 'Kaduna': 24, 'Abia': 25, 'Ekiti': 26,
        'Bayelsa': 27, 'Delta': 28, 'Zamfara': 29, 'Akwa Ibom': 30, 'Nasarawa': 31, 'Taraba': 32,
        'Niger': 33, 'Katsina': 34, 'Lagos': 35
    }

    mtn_device_mapping = {
        '4G Router': 1,
        'Mobile SIM Card': 2,
        '5G Broadband Router': 3,
        'Broadband MiFi': 4
    }

    customer_review_mapping = {
        'Fair': 1,
        'Poor': 2,
        'Good': 3,
        'Excellent': 4,
        'Very Good': 5
    }

    subscription_plan_mapping = {
        '165GB Monthly Plan': 1,
        '12.5GB Monthly Plan': 2,
        '150GB FUP Monthly Unlimited': 3,
        '1GB+1.5mins Daily Plan': 4,
        '30GB Monthly Broadband Plan': 5,
        '10GB+10mins Monthly Plan': 6,
        '25GB Monthly Plan': 7,
        '7GB Monthly Plan': 8,
        '1.5TB Yearly Broadband Plan': 9,
        '65GB Monthly Plan': 10,
        '120GB Monthly Broadband Plan': 11,
        '300GB FUP Monthly Unlimited': 12,
        '60GB Monthly Broadband Plan': 13,
        '500MB Daily Plan': 14,
        '3.2GB 2-Day Plan': 15,
        '20GB Monthly Plan': 16,
        '2.5GB 2-Day Plan': 17,
        '450GB 3-Month Broadband Plan': 18,
        '200GB Monthly Broadband Plan': 19,
        '1.5GB 2-Day Plan': 20,
        '16.5GB+10mins Monthly Plan': 21
    }

    # Map categorical variables to numerical
    Gender_map = Gender_mapping[Gender]
    state_map = state_mapping[State]
    subscription_plan_map = subscription_plan_mapping[Subscription_Plan]
    customer_review_map = customer_review_mapping[Customer_review]
    mtn_device_map = mtn_device_mapping[MTN_Device]

    # Log transform Total Revenue
    Total_Revenue_log = np.log1p(Total_Revenue)

    if st.button('🔮 Predict Customer Churn'):
        features = [
            [
                Age,
                state_map,
                subscription_plan_map,
                customer_review_map,
                mtn_device_map,
                Total_Revenue_log,
                Gender_map,
                Unit_Price,
                Number_of_Times_Purchased,
                tenure,
                Data_usage
            ]
        ]
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        if prediction[0] == 1:
            st.markdown(f"🚩 Customer is likely to churn! Probability: {round(probability * 100, 2)}%")
        else:
            st.markdown(f"✅ Customer is unlikely to churn!")
            
            
    st.markdown("---")
            
        # Load dataset
        
    st.markdown("## MTN Customer Dataset")
    df = pd.read_csv('mtn_customer_churn.csv')
    st.dataframe(df)


# Visualization in tab2
with tab2:
    
    #Age Distribution
    st.markdown("## Age Distribution")
    plt.figure(figsize=(5,3))
    sns.histplot(df['Age'].dropna(), bins=20)
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    st.pyplot(plt)

    st.markdown("---")

    # 3. Geographic Distribution
    st.markdown("## Customers State")
    plt.figure(figsize=(33,14))
    sns.countplot(data=df, x='State')
    plt.title('Customer Distribution by State')
    plt.xlabel('State')
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.ylabel('Number of Customers')
    st.pyplot(plt)

    st.markdown("---")



    # 4. Device Usage
    st.markdown("## Device Usage")
    plt.figure(figsize=(10,3))
    sns.countplot(data=df, x='MTN Device')
    plt.title('Device Usage')
    plt.xlabel('Device Type')
    plt.ylabel('Count')
    st.pyplot(plt)

    st.markdown("---")

    # 5. Gender Distribution
    st.markdown("## Gender Distribution")
    plt.figure(figsize=(5,3))
    sns.countplot(data=df, x='Gender')
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    st.pyplot(plt)

    st.markdown("---")

    # 6. Satisfaction Rate
    st.markdown("## Satisfaction Rate")
    plt.figure(figsize=(5,3))
    sns.histplot(df['Satisfaction Rate'].dropna(), bins=20)
    plt.title('Customer Satisfaction Rate')
    plt.xlabel('Satisfaction Rate')
    plt.ylabel('Count')
    st.pyplot(plt)

    st.markdown("---")

    # 7. Customer Tenure
    st.markdown("## Customer Tenure")
    plt.figure(figsize=(5,3))
    sns.histplot(df['Customer Tenure in months'].dropna(), bins=20)
    plt.title('Customer Tenure Distribution')
    plt.xlabel('Months')
    plt.ylabel('Count')
    st.pyplot(plt)


    st.markdown("---")

    #8. Subscription Plans
    st.markdown("## Subscription Plans")
    plt.figure(figsize=(12,5))
    sns.countplot(data=df, x='Subscription Plan')
    plt.title('Subscription Plan Popularity')
    plt.xlabel('Plan')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.markdown("---")

    # 10. Churn Analysis
    st.markdown("## Churn Analysis")
    churn_counts = df['Customer Churn Status'].value_counts()
    plt.figure(figsize=(5,3))
    sns.barplot(x=churn_counts.index, y=churn_counts.values)
    plt.title('Customer Churn Status')
    plt.xlabel('Churned')
    plt.ylabel('Count')
    st.pyplot(plt)

    st.markdown("---")

    # 11. Reasons for Churn (if categorical)
    st.markdown("## Reasons for Churn")
    plt.figure(figsize=(5,3))
    sns.countplot(data=df, y='Reasons for Churn', order=df['Reasons for Churn'].value_counts().index)
    plt.title('Reasons for Customer Churn')
    plt.xlabel('Count')
    plt.ylabel('Reason')
    st.pyplot(plt)

