# Core Libraries
import pandas as pd
import numpy as np

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning - Preprocessing & Model
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# XGBoost
import xgboost as xgb

# Explainable AI
import shap
import lime
import lime.lime_tabular

# Streamlit (for the app)
import streamlit as st

# --- Configuration: USER INPUT REQUIRED (Verify these after loading) ---
# 1. CSV File Path:
#     This is set to the name of the uploaded file.
#     Ensure this file is in the same directory as your Python script, or provide the full path.
CSV_FILE_PATH = 'Maternal Health Risk Data Set.csv' # <--- SET TO YOUR UPLOADED FILE

# 2. Target Variable Column Name:
#     The name of the column in your CSV file that you want to predict.
#     Inspect your CSV's columns to confirm this is correct.
TARGET_COLUMN = 'RiskLevel'  # <--- !!! VERIFY THIS IS CORRECT FOR YOUR CSV !!!

# 3. Unique Identifier Column (Optional, for dropping):
#     If you have a column like 'PatientID' that should not be used for training.
ID_COLUMN = None # e.g., 'PatientID' <--- !!! CHANGE THIS IF YOU HAVE ONE !!!
# --- End of Configuration ---

# Set some global display options
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.max_open_warning'] = 0 # Suppress too many open figures warning

# --- 1. Data Loading and Initial Inspection ---
@st.cache_data # Cache data loading for Streamlit efficiency
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        st.success(f"Dataset loaded successfully from '{file_path}'")

        # Automatically print head and info to help user verify columns
        st.subheader("Initial Data Preview (from CSV)")
        st.write("First 5 rows of the loaded data:")
        st.dataframe(df.head().astype(str))

        # Display clean table of column info
# Display clean table of column info
        st.subheader("DataFrame Info (Structured View)")
        df_info_table = pd.DataFrame({
            "Column Name": df.columns,
            "Non-Null Count": df.notnull().sum().values,
            "Data Type": df.dtypes.values
        })
        st.dataframe(df_info_table.astype(str), use_container_width=True)




        if ID_COLUMN and ID_COLUMN in df.columns:
            df_cleaned = df.drop(columns=[ID_COLUMN])
            st.info(f"Dropped ID column: '{ID_COLUMN}'")
        else:
            df_cleaned = df.copy() # Work with a copy

        if TARGET_COLUMN not in df_cleaned.columns:
            st.error(f"FATAL: Target column '{TARGET_COLUMN}' not found in the dataset!")
            st.error(f"Available columns are: {df_cleaned.columns.tolist()}")
            st.error("Please update the TARGET_COLUMN variable in the script and rerun.")
            return None # Stop execution

        return df_cleaned
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct location.")
        return None
    except Exception as e:
        st.error(f"An error occurred while trying to read the CSV file: {e}")
        return None

# --- 2. Exploratory Data Analysis (EDA) ---
def perform_eda(df_eda, target_column_eda): # Use different var names to avoid clashes
    st.header("Exploratory Data Analysis (EDA)")

    st.write("Shape of the dataset (after potential ID drop):", df_eda.shape)
    # df.head() and df.info() already shown during load_data

    st.write("Descriptive Statistics:", df_eda.describe(include='all'))

    missing_vals = df_eda.isnull().sum()
    st.write("Missing Values per column:", missing_vals[missing_vals > 0])
    if missing_vals.sum() == 0:
        st.success("No missing values found in the dataset (post ID drop).")


    if target_column_eda in df_eda.columns:
        st.write(f"Value Counts for Target Variable ('{target_column_eda}'):")
        st.write(df_eda[target_column_eda].value_counts())

        fig_target_dist, ax_target_dist = plt.subplots()
        sns.countplot(x=target_column_eda, data=df_eda, ax=ax_target_dist, order=df_eda[target_column_eda].value_counts().index)
        ax_target_dist.set_title(f'Distribution of {target_column_eda}')
        plt.xticks(rotation=45)
        st.pyplot(fig_target_dist)
        plt.clf()

        # Correlation Heatmap (numerical features only)
        numerical_features_for_corr = df_eda.select_dtypes(include=np.number).columns.tolist()
        if len(numerical_features_for_corr) > 1:
            df_corr_temp = df_eda.copy() # Use a temporary copy for potential encoding
            if df_corr_temp[target_column_eda].dtype == 'object' or pd.api.types.is_categorical_dtype(df_corr_temp[target_column_eda]):
                temp_label_encoder = LabelEncoder()
                # Ensure the encoded column name is unique and clearly temporary
                encoded_target_name_for_corr = target_column_eda + '_encoded_for_corr'
                df_corr_temp[encoded_target_name_for_corr] = temp_label_encoder.fit_transform(df_corr_temp[target_column_eda])
                if encoded_target_name_for_corr not in numerical_features_for_corr :
                    numerical_features_for_corr_updated = numerical_features_for_corr + [encoded_target_name_for_corr]
                else:
                    numerical_features_for_corr_updated = numerical_features_for_corr # Already included
            else: # Target is already numerical
                numerical_features_for_corr_updated = numerical_features_for_corr


            if len(numerical_features_for_corr_updated) >1:
                plt.figure(figsize=(12, 10))
                correlation_matrix = df_corr_temp[numerical_features_for_corr_updated].corr()
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
                plt.title('Correlation Heatmap of Numerical Features (and Encoded Target if applicable)')
                st.pyplot(plt)
                plt.clf()
            else:
                st.write("Not enough numerical features to generate a correlation heatmap.")

        # Box plots for numerical features against the target variable
        # Make sure numerical_features does not include the target if it was originally numerical
        numerical_features_for_boxplot = [col for col in df_eda.select_dtypes(include=np.number).columns if col != target_column_eda]

        for col in numerical_features_for_boxplot:
            if col in df_eda.columns:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=target_column_eda, y=col, data=df_eda, order=df_eda[target_column_eda].value_counts().index)
                plt.title(f'{col} vs. {target_column_eda}')
                plt.xticks(rotation=45)
                st.pyplot(plt)
                plt.clf()
    else:
        st.warning(f"Target column '{target_column_eda}' not found for detailed EDA.")

# --- 3. Preprocessing and Feature Engineering ---
@st.cache_resource # Cache preprocessor and label encoder
def get_preprocessor_and_label_encoder(df_for_setup, target_column_setup):
    X_setup = df_for_setup.drop(target_column_setup, axis=1)
    y_setup = df_for_setup[target_column_setup]

    le = LabelEncoder()
    le.fit(y_setup) # Fit label encoder on the full target column
    target_mapping = {index: label for index, label in enumerate(le.classes_)}
    st.write("Target Variable Encoding Mapping:", target_mapping)

    numerical_cols_setup = X_setup.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_setup = X_setup.select_dtypes(include=['object', 'category']).columns.tolist()

    st.write(f"Identified Numerical Columns for Preprocessing: {numerical_cols_setup}")
    st.write(f"Identified Categorical Columns for Preprocessing: {categorical_cols_setup}")

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor_obj = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols_setup),
        ('cat', categorical_pipeline, categorical_cols_setup)
    ], remainder='passthrough')

    # Fit the preprocessor on a sample of X (or all X if not too large) to get feature names
    # This is to ensure get_feature_names_out works correctly later
    preprocessor_obj.fit(X_setup)
    try:
        if categorical_cols_setup:
            ohe_feature_names = preprocessor_obj.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols_setup)
            processed_feature_names_list = numerical_cols_setup + list(ohe_feature_names)
        else:
            processed_feature_names_list = numerical_cols_setup.copy()

        if preprocessor_obj.remainder == 'passthrough':
            # Logic to get names of remainder columns
            processed_cols_flat = [item for sublist in [numerical_cols_setup, categorical_cols_setup] for item in sublist]
            remainder_cols = [col for col in X_setup.columns if col not in processed_cols_flat]
            processed_feature_names_list.extend(remainder_cols)

    except Exception:
        st.warning("Could not reliably get all OHE feature names. Using generic names if issues persist.")
        # Fallback: generate generic names based on shape after transformation
        # This part will be handled properly during the actual transform
        num_cat_features_transformed = 0
        if categorical_cols_setup: # Estimate number of OHE features
            temp_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            temp_ohe.fit(X_setup[categorical_cols_setup].fillna(X_setup[categorical_cols_setup].mode().iloc[0]))
            num_cat_features_transformed = temp_ohe.get_feature_names_out(categorical_cols_setup).shape[0]
        else: # no categorical columns
             num_cat_features_transformed = 0

        processed_feature_names_list = numerical_cols_setup + [f"cat_{i}" for i in range(num_cat_features_transformed)]
        # Add placeholder for remainder if any (less likely with explicit num/cat)


    return preprocessor_obj, le, numerical_cols_setup, categorical_cols_setup, processed_feature_names_list


def preprocess_data_split(df_to_split, target_column_split, preprocessor, label_encoder_obj, processed_feature_names_global):
    df_copy = df_to_split.copy()
    df_copy[target_column_split] = label_encoder_obj.transform(df_copy[target_column_split])

    X_split = df_copy.drop(target_column_split, axis=1)
    y_split = df_copy[target_column_split]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_split, y_split, test_size=0.25, random_state=42, stratify=y_split
    )

    X_train = preprocessor.transform(X_train_raw) # Already fitted
    X_test = preprocessor.transform(X_test_raw)

    # Use the globally determined processed_feature_names
    # Ensure the number of columns matches the number of feature names
    if X_train.shape[1] != len(processed_feature_names_global):
        st.warning(f"Mismatch in transformed columns ({X_train.shape[1]}) and feature names ({len(processed_feature_names_global)}). Re-evaluating names.")
        # Fallback: recreate names based on actual transformed shape (less ideal but a failsafe)
        num_numerical = len(preprocessor.transformers_[0][2]) # Assuming first is num
        num_transformed_cols = X_train.shape[1]
        num_ohe_plus_remainder = num_transformed_cols - num_numerical
        current_numerical_cols = preprocessor.transformers_[0][2]
        new_feature_names = current_numerical_cols + [f"ohe_rem_{i}" for i in range(num_ohe_plus_remainder)]

        if len(new_feature_names) == X_train.shape[1]:
             processed_feature_names_final = new_feature_names
        else: # If still mismatch, use totally generic names
            processed_feature_names_final = [f"feature_{i}" for i in range(X_train.shape[1])]
        st.write("Using fallback feature names:", processed_feature_names_final)
    else:
        processed_feature_names_final = processed_feature_names_global


    X_train_df = pd.DataFrame(X_train, columns=processed_feature_names_final, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test, columns=processed_feature_names_final, index=X_test_raw.index)

    return X_train_df, X_test_df, y_train, y_test, processed_feature_names_final


# --- 4. Model Training and Tuning (XGBoost) ---
@st.cache_resource # Cache the trained model
def train_model(_X_train_model, _y_train_model): # Use different var names
    st.subheader("Model Training (XGBoost with GridSearchCV)")
    model_xgb = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    }
    cv_strat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Reduced splits for speed
    grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid,
                                scoring='accuracy', cv=cv_strat, verbose=1, n_jobs=-1)
    st.write("Starting GridSearchCV... This may take a few minutes.")
    grid_search.fit(_X_train_model, _y_train_model)
    st.write("Best Parameters found by GridSearchCV:", grid_search.best_params_)
    st.write("Best Cross-validation Score (Accuracy):", f"{grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

# --- 5. Model Evaluation ---
def evaluate_model(model_eval, X_test_eval, y_test_eval, label_encoder_eval): # Use different var names
    st.subheader("Model Evaluation")
    y_pred = model_eval.predict(X_test_eval)
    y_pred_proba = model_eval.predict_proba(X_test_eval)

    accuracy = accuracy_score(y_test_eval, y_pred)
    # average='weighted' for multiclass problems
    precision = precision_score(y_test_eval, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_eval, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_eval, y_pred, average='weighted', zero_division=0)

    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision (weighted): {precision:.4f}")
    st.write(f"Recall (weighted): {recall:.4f}")
    st.write(f"F1-score (weighted): {f1:.4f}")

    num_classes = len(np.unique(y_test_eval))
    if num_classes > 2 : # Multiclass
        try:
            roc_auc = roc_auc_score(y_test_eval, y_pred_proba, multi_class='ovr', average='weighted')
            st.write(f"ROC AUC (weighted OvR): {roc_auc:.4f}")
        except ValueError as e_auc: # Handle cases where AUC cannot be computed
             st.warning(f"Could not calculate ROC AUC directly for multiclass: {e_auc}. Probabilities might be an issue or only one class present in y_true.")
        except Exception as e:
            st.warning(f"An unexpected error occurred calculating ROC AUC: {e}")

    elif num_classes == 2: # Binary
        try:
            roc_auc = roc_auc_score(y_test_eval, y_pred_proba[:, 1]) # Probability of positive class
            st.write(f"ROC AUC: {roc_auc:.4f}")
        except Exception as e:
            st.warning(f"Could not calculate ROC AUC for binary: {e}")
    else: # Single class in y_test, ROC AUC not meaningful
        st.warning("ROC AUC not calculated as only one class is present in the test set labels.")


    st.text("Classification Report:")
    class_names_eval = label_encoder_eval.classes_.astype(str)
    report = classification_report(y_test_eval, y_pred, target_names=class_names_eval, zero_division=0, output_dict=False)
    st.text(report)

    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test_eval, y_pred, labels=label_encoder_eval.transform(class_names_eval)) # Ensure labels order
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_eval, yticklabels=class_names_eval, ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)
    plt.clf()

# --- 6. Explainable AI (XAI) ---
def explain_model(model_xai, X_train_df_xai, X_test_df_xai, feature_names_xai, label_encoder_xai): # Use diff var names
    st.subheader("Explainable AI (XAI)")

    if not isinstance(X_train_df_xai, pd.DataFrame) or not all(X_train_df_xai.columns == feature_names_xai):
        X_train_df_xai = pd.DataFrame(X_train_df_xai, columns=feature_names_xai)
    if not isinstance(X_test_df_xai, pd.DataFrame) or not all(X_test_df_xai.columns == feature_names_xai):
        X_test_df_xai = pd.DataFrame(X_test_df_xai, columns=feature_names_xai)


    st.markdown("#### SHAP Explanations")
    try:
        explainer_shap = shap.TreeExplainer(model_xai, X_train_df_xai)
        shap_values_list = explainer_shap.shap_values(X_test_df_xai)

        st.write("SHAP Summary Plot (Global Feature Importance - Bar):")
        fig_shap_bar, ax_shap_bar = plt.subplots() # Use explicit axes
        shap.summary_plot(shap_values_list, X_test_df_xai, plot_type="bar", class_names=label_encoder_xai.classes_.astype(str), feature_names=feature_names_xai, show=False, plot_size=None)
        plt.tight_layout()
        st.pyplot(fig_shap_bar)
        plt.clf()


        # For beeswarm, typically shown per class for multiclass
        # Here, we show for the first class as an example. In a real app, you might let the user choose.
        if isinstance(shap_values_list, list) and len(shap_values_list) > 0:
            class_index_for_beeswarm = 0 # Example: first class
            st.write(f"SHAP Beeswarm Plot (for class: {label_encoder_xai.classes_[class_index_for_beeswarm]}):")
            fig_shap_beeswarm, ax_shap_beeswarm = plt.subplots()
            shap.summary_plot(shap_values_list[class_index_for_beeswarm], X_test_df_xai, feature_names=feature_names_xai, show=False, plot_size=None)
            plt.title(f"SHAP Beeswarm for class: {label_encoder_xai.classes_[class_index_for_beeswarm]}")
            plt.tight_layout()
            st.pyplot(fig_shap_beeswarm)
            plt.clf()
        else: # Binary case (not expected with multi:softprob, but good to handle)
            st.write("SHAP Beeswarm Plot:")
            fig_shap_beeswarm, ax_shap_beeswarm = plt.subplots()
            shap.summary_plot(shap_values_list, X_test_df_xai, feature_names=feature_names_xai, show=False, plot_size=None)
            plt.tight_layout()
            st.pyplot(fig_shap_beeswarm)
            plt.clf()


        st.write("SHAP Force Plot (Local Explanation for first test instance):")
        idx_to_explain = 0
        expected_value_shap = explainer_shap.expected_value
        # ...existing code...
        if isinstance(shap_values_list, list): # Multiclass
            predicted_class_idx_shap = model_xai.predict(X_test_df_xai.iloc[[idx_to_explain]])[0]
            st.write(f"Explaining prediction for class: {label_encoder_xai.classes_[predicted_class_idx_shap]}")
            # MODIFICATION: Changed shap.force_plot to shap.plots.force
            force_plot_html = shap.plots.force(expected_value_shap[predicted_class_idx_shap],
                                            shap_values_list[predicted_class_idx_shap][idx_to_explain,:],
                                            X_test_df_xai.iloc[idx_to_explain,:],
                                            link="logit", # Use logit for probabilities
                                            matplotlib=False) # Output HTML
            st.components.v1.html(force_plot_html.html(), height=200, scrolling=True) # Adjust height as needed
# ...existing code... # Adjust height as needed

            st.write("SHAP Waterfall Plot (Local Explanation for first test instance):")
            fig_waterfall, ax_waterfall = plt.subplots()
            shap.waterfall_plot(shap.Explanation(values=shap_values_list[predicted_class_idx_shap][idx_to_explain],
                                                base_values=expected_value_shap[predicted_class_idx_shap],
                                                data=X_test_df_xai.iloc[idx_to_explain].values, # Ensure it's numpy array for Explanation
                                                feature_names=feature_names_xai),
                                    max_display=15, show=False)
            plt.tight_layout()
            st.pyplot(fig_waterfall)
            plt.clf()
        # ... (binary case for SHAP local if needed) ...

    except Exception as e_shap:
        st.error(f"Error generating SHAP explanations: {e_shap}")
        import traceback
        st.text(traceback.format_exc())

    st.markdown("#### LIME Explanations")
    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_df_xai.values,
            feature_names=feature_names_xai,
            class_names=label_encoder_xai.classes_.astype(str),
            mode='classification',
            discretize_continuous=True
        )
        idx_to_explain_lime = 0
        instance_to_explain_lime = X_test_df_xai.iloc[idx_to_explain_lime].values
        explanation_lime = lime_explainer.explain_instance(
            data_row=instance_to_explain_lime,
            predict_fn=model_xai.predict_proba,
            num_features=10,
            top_labels=len(label_encoder_xai.classes_)
        )
        predicted_class_for_lime = label_encoder_xai.inverse_transform(model_xai.predict(X_test_df_xai.iloc[[idx_to_explain_lime]]))[0]
        st.write(f"LIME Explanation for first test instance (Predicted Class: {predicted_class_for_lime}):")
        for i, class_name_lime in enumerate(label_encoder_xai.classes_):
            if i in explanation_lime.available_labels():
                st.write(f"Explanation for class: {class_name_lime}")
                fig_lime = explanation_lime.as_pyplot_figure(label=i)
                st.pyplot(fig_lime)
                plt.clf()
    except Exception as e_lime:
        st.error(f"Error generating LIME explanations: {e_lime}")
        import traceback
        st.text(traceback.format_exc())

# --- Main Streamlit App Logic ---
# --- Main Streamlit App Logic ---
def main():
    st.set_page_config(layout="wide", page_title="Maternal Health Risk Prediction & XAI")
    st.title("HerShield AI")

    st.sidebar.header("Project Controls")
    if st.sidebar.button("Reload Data and Process"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    df_original = load_data(CSV_FILE_PATH)

    if df_original is not None:
        if TARGET_COLUMN not in df_original.columns: # Additional check post-loading, if load_data didn't stop
            st.error(f"Target column '{TARGET_COLUMN}' re-check failed. Please ensure it's correct in config and CSV.")
            return

        run_eda_opt = st.sidebar.checkbox("Show Exploratory Data Analysis (EDA)", True)
        if run_eda_opt:
            perform_eda(df_original.copy(), TARGET_COLUMN)

        st.header("1. Data Preprocessing Setup")
        preprocessor_obj, label_encoder_obj, numerical_cols, categorical_cols, processed_feature_names_initial = get_preprocessor_and_label_encoder(df_original, TARGET_COLUMN)

        if preprocessor_obj and label_encoder_obj:
            st.header("2. Data Splitting & Transformation")
            X_train_df, X_test_df, y_train, y_test, final_feature_names = preprocess_data_split(
                df_original, TARGET_COLUMN, preprocessor_obj, label_encoder_obj, processed_feature_names_initial
            )
            st.success("Data successfully preprocessed and split.")
            st.write(f"Training data shape: {X_train_df.shape}, Test data shape: {X_test_df.shape}")
            st.write(f"Number of features after processing: {len(final_feature_names)}")

            if X_train_df is not None and not X_train_df.empty:
                st.header("3. Model Training & Tuning")
                trained_model = train_model(X_train_df, y_train)

                st.header("4. Model Evaluation")
                evaluate_model(trained_model, X_test_df, y_test, label_encoder_obj)

                st.header("5. Explainable AI (XAI)")
                explain_model(trained_model, X_train_df, X_test_df, final_feature_names, label_encoder_obj)

                # --- Prediction on New Data (Sidebar) ---
                st.sidebar.divider()
                st.sidebar.header("Test Single Prediction")
                st.sidebar.write("Input features for a new prediction:")
                input_data = {}
                original_cols_for_input_sidebar = [col for col in df_original.columns if col != TARGET_COLUMN and (ID_COLUMN is None or col != ID_COLUMN)]

                for col_sidebar in original_cols_for_input_sidebar:
                    if col_sidebar in numerical_cols:
                        default_val_sidebar = df_original[col_sidebar].median() if not pd.isna(df_original[col_sidebar].median()) else 0.0
                        input_data[col_sidebar] = st.sidebar.number_input(f"Enter {col_sidebar}", value=float(default_val_sidebar), format="%.2f", key=f"input_{col_sidebar}")
                    elif col_sidebar in categorical_cols:
                        unique_vals_sidebar = df_original[col_sidebar].dropna().unique().tolist()
                        default_cat_val_sidebar = df_original[col_sidebar].mode()[0] if not df_original[col_sidebar].mode().empty else (unique_vals_sidebar[0] if unique_vals_sidebar else "")
                        if unique_vals_sidebar:
                            input_data[col_sidebar] = st.sidebar.selectbox(f"Select {col_sidebar}", options=unique_vals_sidebar, index=unique_vals_sidebar.index(default_cat_val_sidebar) if default_cat_val_sidebar in unique_vals_sidebar else 0,  key=f"input_{col_sidebar}")
                        else:
                            input_data[col_sidebar] = st.sidebar.text_input(f"Enter {col_sidebar}", value="", key=f"input_{col_sidebar}")
                    else:
                        input_data[col_sidebar] = st.sidebar.text_input(f"Enter {col_sidebar} (type: {df_original[col_sidebar].dtype})", value="", key=f"input_{col_sidebar}")


                if st.sidebar.button("Predict Maternal Risk", key="predict_button"):
                    try:
                        input_df_raw_sidebar = pd.DataFrame([input_data])
                        st.sidebar.write("Raw Input:", input_df_raw_sidebar)
                        input_processed_array_sidebar = preprocessor_obj.transform(input_df_raw_sidebar)
                        input_processed_df_sidebar = pd.DataFrame(input_processed_array_sidebar, columns=final_feature_names)

                        prediction_encoded_sidebar = trained_model.predict(input_processed_df_sidebar)
                        prediction_proba_sidebar = trained_model.predict_proba(input_processed_df_sidebar)
                        predicted_risk_level_sidebar = label_encoder_obj.inverse_transform(prediction_encoded_sidebar)
                        predicted_class_name = predicted_risk_level_sidebar[0] # Get the string name of the predicted class

                        st.sidebar.subheader("Prediction Result:")
                        st.sidebar.success(f"Predicted Risk Level: **{predicted_class_name}**")
                        st.sidebar.write("Prediction Probabilities:")
                        proba_df_sidebar = pd.DataFrame(prediction_proba_sidebar, columns=label_encoder_obj.classes_)
                        st.sidebar.dataframe(proba_df_sidebar)

                        # --- MODIFIED SECTION FOR CLASS-SPECIFIC STATIC EXPLANATION ---
                        st.sidebar.subheader(f"Shap AI's Explaination for the'{predicted_class_name}' Predictiions")

                        # Define your static explanations here.
                        # IMPORTANT: The keys MUST match the class names from your 'RiskLevel' column.
                        # Example: If your classes are 'low risk', 'mid risk', 'high risk'
                        static_explanations = {
                            "low risk": """
                            **Prediction: Low Risk**

                            A "Low Risk" prediction suggests that, based on the input data, the individual currently exhibits characteristics associated with a lower likelihood of maternal health complications.

                            **General Considerations for Low Risk:**
                            * Continue regular prenatal check-ups as advised by your healthcare provider.
                            * Maintain a healthy lifestyle, including a balanced diet and appropriate exercise.
                            * Be aware of any new or changing symptoms and report them to your doctor.
                            * Factors like stable blood pressure, normal blood sugar, and absence of other acute symptoms typically contribute to this assessment.

                            *This is a general interpretation. Always consult with a healthcare professional for personalized medical advice.*
                            """,
                            "mid risk": """
                            **Prediction: Mid Risk** (or Moderate Risk)

                            A "Mid Risk" prediction indicates that some input factors suggest a moderate potential for maternal health complications. This warrants closer attention and possibly more frequent monitoring.

                            **General Considerations for Mid Risk:**
                            * It's crucial to discuss these results thoroughly with your healthcare provider.
                            * Your doctor may recommend more frequent monitoring, lifestyle adjustments, or specific tests.
                            * Factors such as slightly elevated blood pressure, borderline blood sugar levels, or a combination of less severe individual risk factors might lead to this classification.
                            * Adherence to medical advice is key to managing this risk level.

                            *This is a general interpretation. Always consult with a healthcare professional for personalized medical advice.*
                            """,
                            "high risk": """
                            **Prediction: High Risk**

                            A "High Risk" prediction is a serious indication that the input data shows significant factors associated with a higher likelihood of maternal health complications. Immediate and careful medical management is essential.

                            **General Considerations for High Risk:**
                            * **Consult your healthcare provider immediately.** This prediction highlights the need for urgent medical attention and a tailored management plan.
                            * Factors such as significantly high blood pressure, very high blood sugar, advanced maternal age combined with other conditions, or a history of severe complications often contribute to this assessment.
                            * Specialized care, intensive monitoring, and potentially interventions may be required.

                            *This is a general interpretation and not a diagnosis. Prioritize consultation with a healthcare professional for personalized medical advice and treatment.*
                            """
                            # Add more classes and their explanations if your dataset has them.
                            # For example:
                            # "very high risk": """..."""
                        }

                        explanation_to_show = static_explanations.get(predicted_class_name, "A general explanation for this predicted risk level is not available. The model has assessed the risk based on the input features. Please consult the global XAI plots for general feature importance and discuss with a healthcare professional.")
                        st.sidebar.markdown(explanation_to_show)
                        # --- END OF MODIFIED SECTION ---

                    except Exception as e_pred_sidebar:
                        st.sidebar.error(f"Error during prediction: {e_pred_sidebar}")
                        import traceback
                        st.sidebar.text(traceback.format_exc())
            else:
                st.error("Training data is empty or not processed correctly. Model training and subsequent steps cannot proceed.")
        else:
            st.error("Preprocessing setup failed. Cannot proceed with model training.")
    else:
        st.error("Data loading failed. Please check the CSV file path and its content.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by Kailash And Lal.")

if __name__ == '__main__':
    shap.initjs()
    main()