import io
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.colab import files

class PlottingMethods:
    """
    Utility class dedicated to generating granular, highly formatted charts
    using Plotly, returning them as HTML-wrapped components for modular embedding.
    """
    
    @staticmethod
    def return_html(fig):
        """Helper to wrap a plotly figure object into an HTML string representation."""
        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    @classmethod
    def generate_bar_chart(cls, df, x_col, y_col, title="Bar Chart", barmode="group", text_auto=True):
        """Generates a standalone Bar Chart."""
        fig = px.bar(df, x=x_col, y=y_col, title=title, barmode=barmode, text_auto=text_auto)
        fig.update_layout(template="plotly_white")
        return cls.return_html(fig)

    @classmethod
    def generate_pie_chart(cls, df, names_col, values_col=None, title="Pie Chart"):
        """Generates a standalone Pie Chart."""
        fig = px.pie(df, names=names_col, values=values_col, title=title, hole=0.3)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(template="plotly_white")
        return cls.return_html(fig)

    @classmethod
    def generate_histogram(cls, df, col, nbins=30, title="Histogram"):
        """Generates a standalone Histogram."""
        fig = px.histogram(df, x=col, nbins=nbins, title=title, marginal="box")
        fig.update_layout(template="plotly_white")
        return cls.return_html(fig)


class DataInspector:
    """
    Main Data Engine class for data ingestion, advanced sanitization,
    imputation, feature engineering prep, and advanced interactive visualizations.
    """
    def __init__(self):
        self.df = None
        self.garbage_strings = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', '']

    # ---------------------------------------------------------
    # 1. Data Ingestion & Sanitization
    # ---------------------------------------------------------
    def upload_data(self):
        """Handles local file uploads directly inside a Google Colab notebook environment."""
        print("Please select your CSV file to upload:")
        uploaded = files.upload()
        if not uploaded:
            print("No file uploaded.")
            return None
        
        file_name = list(uploaded.keys())[0]
        # Ingest while converting typical trash/placeholder inputs instantly to NaN
        self.df = pd.read_csv(io.BytesIO(uploaded[file_name]), na_values=self.garbage_strings)
        print(f"\n Successfully loaded '{file_name}' into engine.")
        self._auto_type_correction()
        return self.df

    def _auto_type_correction(self):
        """Force-converts columns to numeric if it doesn't yield an entirely null structural series."""
        if self.df is None: return
        for col in self.df.columns:
            # Try to force conversion to numeric string variants
            converted = pd.to_numeric(self.df[col], errors='coerce')
            # If the column wasn't completely ruined (turned completely into NaNs), commit it
            if not converted.isna().all() and self.df[col].isna().sum() < len(self.df):
                if self.df[col].dtype == 'object': 
                    self.df[col] = converted


    # ---------------------------------------------------------
    # 2. Structural Analysis & Cleaning
    # ---------------------------------------------------------
    def display_summary(self):
        """Displays high-level row/column metadata breakdowns along with a dataset peek."""
        if self.df is None: 
            print("Engine holds no active dataset structure."); return
        
        print("="*60)
        print(f"DATASET STRUCTURE OVERVIEW: {self.df.shape[0]} Rows | {self.df.shape[1]} Columns")
        print("="*60)
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"Numerical Attributes ({len(num_cols)}): {num_cols}")
        print(f"Categorical Attributes ({len(cat_cols)}): {cat_cols}\n")
        print("Missing Values per Column:")
        print(self.df.isna().sum()[self.df.isna().sum() > 0] if self.df.isna().sum().sum() > 0 else "None")
        print("-"*60)
        print("First 20 Data Rows Preview:")
        display(self.df.head(20))

    def handle_missing_values(self, column, strategy='median', fill_value=None):
        """Imputes missing data values dynamically based on selected math/logic strategy."""
        if self.df is None or column not in self.df.columns: return
        
        if strategy == 'mean':
            self.df[column] = self.df[column].fillna(self.df[column].mean())
        elif strategy == 'median':
            self.df[column] = self.df[column].fillna(self.df[column].median())
        elif strategy == 'mode':
            mode_val = self.df[column].mode()
            if not mode_val.empty:
                self.df[column] = self.df[column].fillna(mode_val[0])
        elif strategy == 'constant':
            self.df[column] = self.df[column].fillna(fill_value)
        print(f"Imputation completed for column '{column}' via strategy: {strategy}.")

    def remove_duplicates(self):
        """Prunes exact row duplications matching structural signatures across the framework."""
        if self.df is None: return
        initial_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {initial_count - len(self.df)} exact duplicate rows.")

    def handle_outliers(self, column, action='flag'):
        """Detects outliers via IQR threshold steps. Action can be 'flag' or 'delete'."""
        if self.df is None or column not in self.df.columns: return
        if not np.issubdtype(self.df[column].dtype, np.number):
            print("Outlier handling must target purely numeric structures.")
            return

        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        
        if action == 'flag':
            self.df[f'{column}_outlier'] = outliers_mask.astype(int)
            print(f"Flagged {outliers_mask.sum()} extreme rows inside new column '{column}_outlier'.")
        elif action == 'delete':
            self.df = self.df[~outliers_mask].reset_index(drop=True)
            print(f"Purged {outliers_mask.sum()} outlier rows based on feature column '{column}'.")

    def delete_rows(self, indices_str):
        """Accepts a comma-separated string of structural indexes to drop rows directly."""
        if self.df is None: return
        try:
            indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
            self.df.drop(index=indices, inplace=True, errors='ignore')
            self.df.reset_index(drop=True, inplace=True)
            print(f"Targeted row indices successfully dropped.")
        except Exception as e:
            print(f"Row drop execution exception triggered: {e}")

    def delete_columns(self, columns_str):
        """Accepts a comma-separated string of structural names to drop columns directly."""
        if self.df is None: return
        cols = [x.strip() for x in columns_str.split(',')]
        self.df.drop(columns=cols, inplace=True, errors='ignore')
        print(f"Targeted columns parsed and eliminated.")


    # ---------------------------------------------------------
    # 3. Feature Engineering Preparation (Normalization)
    # ---------------------------------------------------------
    def extract_normalized_numeric_data(self, columns, method='standard'):
        """Transforms continuous values with selectable mathematical scalers."""
        if self.df is None: return pd.DataFrame()
        sub_df = self.df[columns].copy()
        
        for col in columns:
            if method == 'minmax':
                c_min, c_max = sub_df[col].min(), sub_df[col].max()
                sub_df[col] = (sub_df[col] - c_min) / (c_max - c_min + 1e-9)
            elif method == 'standard':
                sub_df[col] = (sub_df[col] - sub_df[col].mean()) / (sub_df[col].std() + 1e-9)
            elif method == 'robust':
                q25, q50, q75 = sub_df[col].quantile(0.25), sub_df[col].quantile(0.50), sub_df[col].quantile(0.75)
                iqr = q75 - q25
                sub_df[col] = (sub_df[col] - q50) / (iqr + 1e-9)
        return sub_df

    def extract_normalized_categorical_data(self, columns, method='onehot'):
        """Encodes discrete string categoricals into numerical vectors."""
        if self.df is None: return pd.DataFrame()
        sub_df = self.df[columns].copy()
        
        if method == 'onehot':
            return pd.get_dummies(sub_df, columns=columns, drop_first=True, dtype=float)
        
        elif method in ['ordinal', 'uniform']:
            for col in columns:
                # Convert categorical text explicitly into mapped enumeration codes
                sub_df[col] = sub_df[col].astype('category').cat.codes.astype(float)
                if method == 'uniform':
                    c_min, c_max = sub_df[col].min(), sub_df[col].max()
                    if c_max > c_min:
                        sub_df[col] = (sub_df[col] - c_min) / (c_max - c_min)
            return sub_df

    def merge_features(self, num_df, cat_df):
        """Assembles treated matrix arrays cleanly back into a working analysis frame."""
        return pd.concat([num_df, cat_df], axis=1)


    # ---------------------------------------------------------
    # 4. Advanced Interactive Visualization (Plotly)
    # ---------------------------------------------------------
    def plot_univariate_subplots(self, column):
        """Generates an intuitive 3-panel continuous profile tracking shape, trend, and spread."""
        if self.df is None or column not in self.df.columns: return
        
        fig = make_subplots(
            rows=3, cols=1, 
            subplot_titles=(f"Violin & Box Plot of {column}", f"Index vs Value Scatter Map", f"Distribution Frequency Histogram"),
            vertical_spacing=0.12
        )
        
        # Panel 1: Box / Violin Combo
        fig.add_trace(go.Violin(x=self.df[column], box_visible=True, meanline_visible=True, name="Distribution", marker_color="mediumpurple"), row=1, col=1)
        # Panel 2: Sequence Tracking Scatter
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df[column], mode='markers', marker=dict(color='teal', opacity=0.6), name="Sequence"), row=2, col=1)
        # Panel 3: Frequency Profile
        fig.add_trace(go.Histogram(x=self.df[column], nbinsx=30, marker_color="crimson", name="Frequency"), row=3, col=1)
        
        fig.update_layout(height=800, title_text=f"Multi-Angle Univariate Report for Attribute: {column}", showlegend=False, template="plotly_white")
        fig.show()

    def plot_relationship(self, col_x, col_y):
        """Detects data types automatically and chooses the correct statistical chart type."""
        if self.df is None or col_x not in self.df.columns or col_y not in self.df.columns: return
        
        is_x_num = np.issubdtype(self.df[col_x].dtype, np.number)
        is_y_num = np.issubdtype(self.df[col_y].dtype, np.number)
        
        # Case A: Continuous X vs Continuous Y (Numeric vs Numeric)
        if is_x_num and is_y_num:
            fig = px.scatter(self.df, x=col_x, y=col_y, trendline="ols", title=f"Scatter Analysis: {col_y} vs {col_x} with OLS Fit Line", template="plotly_white")
            fig.show()
            
        # Case B: Categorical vs Continuous (Mixed Type Layout)
        elif (not is_x_num and is_y_num) or (is_x_num and not is_y_num):
            cat, num = (col_x, col_y) if not is_x_num else (col_y, col_x)
            fig = px.box(self.df, x=cat, y=num, points="all", title=f"Box-Whisker Profile of {num} Split by Factor: {cat}", template="plotly_white")
            fig.show()
            
        # Case C: Categorical vs Categorical
        else:
            counts = self.df.groupby([col_x, col_y]).size().reset_index(name='Observations')
            fig = px.bar(counts, x=col_x, y='Observations', color=col_y, barmode="group", title=f"Comparative Grouped Frequency Distribution: {col_x} & {col_y}", template="plotly_white")
            fig.show()

    def plot_categorical_frequency(self, column):
        """Plots categorical distribution displaying both raw counts and percentage layout labels."""
        if self.df is None or column not in self.df.columns: return
        
        counts = self.df[column].value_counts(dropna=True).reset_index()
        counts.columns = [column, 'Count']
        counts['Percentage'] = (counts['Count'] / counts['Count'].sum() * 100).round(2)
        counts['Label'] = counts['Count'].astype(str) + " (" + counts['Percentage'].astype(str) + "%)"
        
        fig = px.bar(counts, x=column, y='Count', text='Label', title=f"Frequency Summary Analysis for Attribute Categorical Factor: {column}", template="plotly_white")
        fig.update_traces(textposition='outside')
        fig.show()


    # ---------------------------------------------------------
    # 5. Deep Statistical Insights (Unified Heatmap)
    # ---------------------------------------------------------
    def plot_all_associations_heatmap(self):
        """
        Computes a unified matrix across all data types:
        - Numeric vs Numeric: Pearson's r
        - Categorical vs Categorical: Cramér's V
        - Mixed: Eta value derived from an ANOVA framework
        """
        if self.df is None: return
        
        # Clean down any outlier flag columns before tracking association structures
        cols = [c for c in self.df.columns if not c.endswith('_outlier')]
        n = len(cols)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                col1, col2 = cols[i], cols[j]
                if i == j:
                    matrix[i, j] = 1.0; continue
                
                is_1_num = np.issubdtype(self.df[col1].dtype, np.number)
                is_2_num = np.issubdtype(self.df[col2].dtype, np.number)
                
                # Drop null lines local to pair metrics to avoid calculation bias
                pair_df = self.df[[col1, col2]].dropna()
                if len(pair_df) < 5:
                    matrix[i, j] = np.nan; continue
                
                # 1. Pearson's r (Continuous vs Continuous)
                if is_1_num and is_2_num:
                    r, _ = stats.pearsonr(pair_df[col1], pair_df[col2])
                    matrix[i, j] = abs(r)
                
                # 2. Cramér's V (Discrete vs Discrete)
                elif not is_1_num and not is_2_num:
                    confusion_matrix = pd.crosstab(pair_df[col1], pair_df[col2])
                    chi2 = stats.chi2_contingency(confusion_matrix)[0]
                    total_obs = confusion_matrix.sum().sum()
                    phi2 = chi2 / total_obs
                    r, k = confusion_matrix.shape
                    # Standard bias correction mapping for Cramer's calculation
                    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(total_obs-1))
                    rcorr = r - ((r-1)**2)/(total_obs-1)
                    kcorr = k - ((k-1)**2)/(total_obs-1)
                    denom = min((kcorr-1), (rcorr-1))
                    matrix[i, j] = np.sqrt(phi2corr / denom) if denom > 0 else 0.0
                
                # 3. Mixed Mode (Continuous vs Discrete Evaluation via Eta-Squared)
                else:
                    num_col, cat_col = (col1, col2) if is_1_num else (col2, col1)
                    groups = [group[num_col].values for name, group in pair_df.groupby(cat_col)]
                    if len(groups) > 1 and sum(len(g) for g in groups) > len(groups):
                        f_val, _ = stats.f_oneway(*groups)
                        # Derive Eta translation link index via analytical transformation metric
                        df_between = len(groups) - 1
                        df_within = len(pair_df) - len(groups)
                        if (f_val * df_between + df_within) > 0:
                            eta2 = (f_val * df_between) / (f_val * df_between + df_within)
                            matrix[i, j] = np.sqrt(eta2)
                        else: matrix[i, j] = 0.0
                    else: matrix[i, j] = 0.0
                    
        assoc_df = pd.DataFrame(matrix, index=cols, columns=cols).fillna(0.0)
        
        fig = px.imshow(assoc_df, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis", title="Unified Engine Feature Association Matrix (Pearson / Cramér's V / Eta Indicator Values)", labels=dict(color="Strength"))
        fig.show()