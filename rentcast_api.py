"""Rent Cast API Python Helper Classes"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime
import copy
import requests
import sqlite3
import os
import scipy
import json

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

import warnings
warnings.filterwarnings(
    "ignore",
    message="Converting to PeriodArray/Index representation will drop timezone information.",
    category=UserWarning,
)

class RentCastData():

    URL = "https://api.rentcast.io/v1/properties"
    HEADERS = ['id', 'addressLine1', 'addressLine2', 'city', 'state', 
                'zipCode', 'county', 'latitude', 'longitude', 'squareFootage', 
                'lotSize', 'yearBuilt', 'lastSaleDate', 'lastSalePrice', 
                'bedrooms', 'bathrooms', 'propertyType']

    def __init__(self, 
                 address=None, 
                 city=None, 
                 state=None, 
                 zip=None, 
                 limit=500, 
                 offset_lim=1000,
                 start_date=None,
                 status='Inactive',
                 api_key=''):
        
        # Initalizations
        self.headers = {'accept': 'application/json', 'X-Api-Key': api_key}
        self.querystring = {}
        self.data_raw = None
        self.data_processed = None
        self.offset_lim = offset_lim

        # Sale range - calculate days ago since this date
        if start_date is not None:
            today = datetime.date.today()
            sale_range = (today - start_date).days
            self.querystring['saleDateRange'] = sale_range      

        # Add input args to address string
        address_check = address is not None
        if address_check:
            self.querystring['address'] = address

        city_check = city is not None
        if city_check:
            self.querystring['city'] = city

        state_check = state is not None
        if state_check:
            self.querystring['state'] = state.upper()

        zip_check = zip is not None
        if zip_check:
            self.querystring['zip'] = zip

        self.querystring['limit'] = limit

        # If query information was given, get response. Otherwise read db
        if address_check or city_check or state_check or zip_check:
            self.get_response()

    def get_response(self):

        limit = self.querystring['limit']

        # Loop through offsets
        querystring = copy.deepcopy(self.querystring)
        offsets = np.arange(0, self.offset_lim, 1)
        for offset in offsets:
            querystring['offset'] = limit * offset

            # Get query
            print(querystring)
            response = requests.get(self.URL, 
                                    headers=self.headers, 
                                    params=querystring)
            listings = response.json()

            # Check that a response was given
            if not len(listings):
                print("No more listings at offset %i with limit %i" % (offset, limit))
                break

            df_response = self.parse_response(listings)

            # Add query properties to response
            df_response['Offset'] = limit * offset
            df_response['limit'] = limit
            df_response['Querystring'] = str(querystring)

            # Add data to existing data
            self.data_raw = pd.concat([self.data_raw, df_response], ignore_index=True)

    def parse_response(self, listings):

        df_all = None

        # Loop through results and create dataframe
        for listing_loop in listings:

            listing = copy.deepcopy(listing_loop)
            listing_temp = copy.deepcopy(listing)

            for header in listing_temp:
                # if header not in self.HEADERS:
                #     del listing[header]
                # else:
                listing[header] = [listing[header]]

            df = pd.DataFrame.from_dict(listing)
            df_all = pd.concat([df_all, df], ignore_index=True)

        return df_all
    
    def save_to_db(self, db_path, table_name=None):
        """
        Save a pandas DataFrame to an SQLite database.

        Parameters:
            db_path (str): The name of the SQLite database file.
            table_name (str): The name of the table to store the data in.

        Returns:
            None
        """
        # Fix dicts in dataframes
        df = self.fix_dect_cols(self.data_raw)

        # Ensure the database name ends with .db
        if not db_path.endswith('.db'):
            db_path += '.db'

        # Ensure directory exists
        db_folderpath = os.path.dirname(db_path)
        if not os.path.exists(db_folderpath):
            os.makedirs(db_folderpath)
        
        # Create a connection to the SQLite database
        conn = sqlite3.connect(db_path)

        # Set table name
        if table_name is None:

            city = self.querystring['city']
            state = self.querystring['state']

            if city is None or state is None:
                raise ValueError("No table name for database given")

            table_name = city + '_' +state
            table_name = table_name.replace(' ', '_')
            table_name = table_name.lower()
        
        # Save the DataFrame to the SQLite database
        df.to_sql(table_name, conn, if_exists='replace', index=False, method='multi', chunksize=30)
        
        # Close the connection
        conn.close()

    def fix_dect_cols(self, df):
        df_copy = copy.deepcopy(df)
        # Find columns with dict/list values
        dict_cols = [col for col in df_copy.columns 
                    if df_copy[col].apply(lambda x: isinstance(x, (dict, list))).any()]

        # Convert those columns to JSON strings
        for col in dict_cols:
            df_copy[col] = df_copy[col].apply(
                lambda x: json.dumps(x, default=str, allow_nan=False) 
                        if isinstance(x, (dict, list)) else x
            )
        
        return df_copy
    

class RentCastPlotter():

    BASE_FILTERS = [
            ("price_per_sqft", ">", 100),
            ("price_per_sqft", "<", 5000),
        ]

    def __init__(self):
        self.conn = False
        self.data_raw = {}
        self.data_processed = {}

    @classmethod
    def open_db(cls, db_path):
        """
        Open an SQLite database and return the connection object.

        Parameters:
            db_name (str): The name of the SQLite database file.

        Returns:
            sqlite3.Connection: The connection object to the SQLite database.
        """
        # Ensure the database name ends with .db
        if not db_path.endswith('.db'):
            db_path += '.db'
        
        # Create a connection to the SQLite database
        rcp = cls()
        rcp.conn = sqlite3.connect(db_path)

        return rcp
    
    @property
    def data_all(self):
        """Concatenate all city DataFrames into one DataFrame."""
        dfs = []
        for city_state in self.list_all_cities():

            self.read_city(city_state=city_state)
            df_copy = copy.deepcopy(self.data_processed[city_state])
            df_copy["city_key"] = city_state  # track source city
            dfs.append(df_copy)

        return pd.concat(dfs, ignore_index=True)
    
    def _return_table_name(self, city, state):
        """
        Return a table name given a city and state.

        The table name is formed by concatenating the city and state, separated by an underscore.
        Spaces are replaced with underscores and the entire string is converted to lowercase.

        Parameters:
            city (str): The city (e.g. "San Francisco").
            state (str): The state (e.g. "CA").

        Returns:
            str: The table name (e.g. "san_francisco_ca").
        """
        table_name = city + '_' +  state
        table_name = table_name.replace(' ', '_')
        table_name = table_name.lower()
        return table_name
    
    def list_all_cities(self):
        """
        List all cities in the database.

        This method executes a SQL query to find all tables in the database,
        which correspond to cities. The list of table names is then printed
        to the console.

        Returns:
            None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        return [t[0] for t in tables]  # list of table names
                
    def read_city(self, city=None, state=None, city_state=None):

        # Read data from the specified table
        """
        Read the data for a given city and state from the SQLite database.

        Parameters:
            city (str): The city (e.g. "San Francisco").
            state (str): The state (e.g. "CA").

        Returns:
            None
        """
        if city_state is None:
            table_name = self._return_table_name(city, state)
        else:
            table_name = city_state

        if table_name not in list(self.data_processed):

            df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)

            # Get data and process it
            self.data_raw[table_name] = df
            self.data_processed[table_name] = self.process_data(df)
    
    def add_months(self, date, months):
        # Calculate the target month and year
        """
        Add the specified number of months to the given date.

        Parameters:
            date (datetime.date): The date to modify.
            months (int): The number of months to add.

        Returns:
            datetime.date: The modified date.
        """
        target_month = date.month + months
        target_year = date.year + (target_month - 1) // 12
        target_month = (target_month - 1) % 12 + 1

        # Calculate the day
        day = min(date.day, [31,
                            29 if target_year % 4 == 0 and (target_year % 100 != 0 or target_year % 400 == 0) else 28,
                            31, 30, 31, 30, 31, 31, 30, 31, 30, 31][target_month - 1])

        return date.replace(year=target_year, month=target_month, day=day)

    def _months_between(self, date1, date2):
        # Ensure date1 is earlier than date2
        """
        Calculate the number of months between two dates.

        Parameters:
            date1 (datetime.date): The earlier date.
            date2 (datetime.date): The later date.

        Returns:
            int: The number of months between date1 and date2.
        """
        if date1 > date2:
            date1, date2 = date2, date1
        
        # Calculate the difference in years and months
        year_diff = date2.year - date1.year
        month_diff = date2.month - date1.month
        
        # Total number of months
        total_months = year_diff * 12 + month_diff
        
        return total_months

    def process_data(self, df):
        """Add metrics of interest to the dataset"""

        df_temp = copy.deepcopy(df)

        for col in ["lastSalePrice", "squareFootage"]:
            df_temp[col] = pd.to_numeric(df_temp[col], errors="coerce")

        df_temp = df_temp.dropna(subset=['lastSalePrice', 'squareFootage', 'lastSaleDate'])

        # Price per Square Foot
        df_temp['price_per_sqft'] = df_temp['lastSalePrice'] / df_temp['squareFootage']

        # Duration on market
        df_temp = self.parse_sale_date(df_temp)

        return df_temp

    def parse_sale_date(self, df):
        """Parse last sale date into more readable terms"""

        # Parse once, vectorized
        dt = pd.to_datetime(df["lastSaleDate"], utc=True, errors="coerce")

        # If t0 is naive, make it UTC to match
        t0 = pd.Timestamp("1985-01-01", tz="UTC")

        # Calculate delta
        year_delta = (dt.dt.year - t0.year) * 12
        month_delta = dt.dt.month - t0.month
        months = year_delta + month_delta

        df = df.assign(
            datetime=dt,
            **{"month-year": dt.dt.strftime("%m-%Y")},
            months=months
        )

        return df
    
    def plot_city_states(self, city_states, filters=None):
        """
        Plot a collection of city-states with optional monthly average filtering and overlaying onto an existing figure.

        Parameters
        ----------
        city_states : list of tuples
            List of tuples containing the city and state (e.g. ('San Francisco', 'CA')).
        """
        fig = go.Figure()
        for (city, state) in city_states:
            self.plot_city_state(city, state, avg_only=True, fig=fig, filters=filters)
        # plt.legend(loc='best')
        fig.show()
    
    def plot_city_state(self, city, state, avg_only=False, fig=None, filters=None):

        """
        Plot a city-state with optional monthly average filtering and overlaying onto an existing figure.

        Parameters
        ----------
        city : str
            The city (e.g. "San Francisco").
        state : str
            The state (e.g. "CA").
        avg_only : bool, optional
            If True, only plot the monthly average; if False, plot all samples and monthly average.
            Default is False.
        fig : go.Figure, optional
            If not None, overlay onto this figure; otherwise create a new figure.

        Returns
        -------
        None
        """
        self.read_city(city, state)
        df = self.data_processed[self._return_table_name(city, state)]

        df = self.apply_filters(df, filters) 

        fig = self.plot_processed_trace_ptly(df, avg_only=avg_only, fig=fig)

        return fig

    # def plot_city_state(self, city, state, avg_only=False, fig=None, filter1,=None):

    #     """
    #     """
    #     self.read_city(city, state)
    #     df = self.data_processed[self._return_table_name(city, state)]

    #     df = self.apply_filters(df, filters) 

    #     self.plot_processed_trace_ptly(df, avg_only=avg_only, fig=fig)

    #     # plt.title('Price per SQFT - %s, %s' % (city, state))

    def apply_filters(self, df, filters=None):
        """
        Apply a list of filters to a DataFrame.
        Each filter should be a tuple (column, op, value).
        Supported ops: '>', '<', '>=', '<=', '==', '!='
        """
        if filters is None:
            filters = []

        # Add base filters    
        filters += self.BASE_FILTERS

        for col, op, val in filters:
            if op == ">":
                df = df[df[col] > val]
            elif op == "<":
                df = df[df[col] < val]
            elif op == ">=":
                df = df[df[col] >= val]
            elif op == "<=":
                df = df[df[col] <= val]
            elif op == "==":
                df = df[df[col] == val]
            elif op == "!=":
                df = df[df[col] != val]
            else:
                raise ValueError(f"Unsupported operator: {op}")
        return df

    def plot_processed_trace_mpl(self, df, filter_avg=True, avg_only=False):

        """
        Plot a time series of price per sqft for a city and state.
        The raw data points are plotted as semi-transparent points.
        The average is plotted as a line. If `filter_avg` is True, the
        average is preprocessed to remove spikes.
        If `avg_only` is True, only the average is plotted.
        """
        x = df['months'].values / 12
        y = df['price_per_sqft'].values

        # Averages    
        grouped_df = df.groupby('months')['price_per_sqft'].mean().reset_index()
        x_avg = grouped_df['months'].values / 12
        y_avg = grouped_df['price_per_sqft'].values
        if filter_avg:
            y_avg = self.despike_median(y_avg, 8)

        if avg_only:
            avg_color = None
            avg_lw = 2
        else:
            avg_color = 'black'
            avg_lw = 1

        if not avg_only:
            plt.figure(figsize=(10, 4))
            plt.semilogy(x, y, lw=0, ms=2, marker='o', alpha=0.2)
        plt.semilogy(x_avg, y_avg, lw=avg_lw, ms=0, marker='o', color=avg_color, alpha=0.8)
        plt.ylim([50, 2000])

        # x labels
        int_labels = np.arange(0, 42, 2.5)
        n = len(int_labels)
        lsd = df['lastSaleDate'].values[0]
        dt = datetime.datetime.strptime(lsd, '%Y-%m-%dT%H:%M:%S.%fZ')
        dt_labels = [self.add_months(dt, x * 30) for x in range(n)]
        dt_labels = [dt_label.strftime('%m/%Y') for dt_label in dt_labels]
        plt.xticks(int_labels, dt_labels, rotation=45)
        plt.yticks([10, 30, 100, 300, 1000], [10, 30, 100, 300, 1000])

        plt.title('Price per SQFT - Redwood City')
        plt.ylabel('Price per SQFT ($)')
        plt.xlabel('Time')
        plt.grid(True)


    def plot_processed_trace_ptly(self, df, filter_avg=True, avg_only=False, fig=None):
        """
        Plot price_per_sqft vs lastSaleDate using Plotly with a date x-axis.
        - If avg_only=False: plots all samples (scatter) + monthly average (line).
        - If avg_only=True: adds only the monthly average; pass an existing `fig` to overlay.
        - If `fig` is None, a new Figure is created and returned.
        """
        # ---- Labels / title ----
        city = str(df["city"].iloc[0]).title()
        state = str(df["state"].iloc[0]).upper()
        if not avg_only:
            label = "Average"
            title = f"Price per SQFT - {city}, {state}"
        else:
            label = f"{city}, {state}"
            title = "Price per SQFT"

        # ---- Parse dates, clean data ----
        dt = pd.to_datetime(df["lastSaleDate"], utc=True, errors="coerce")
        y_all = pd.to_numeric(df["price_per_sqft"], errors="coerce")
        mask = dt.notna() & y_all.notna()
        dt = dt[mask]
        y_all = y_all[mask]

        # ---- Monthly average (calendar months) ----
        g = (
            pd.DataFrame({"dt": dt, "pps": y_all})
            .groupby(dt.dt.to_period("M"))["pps"]
            .mean()
        )
        x_avg = g.index.to_timestamp()  # start of month
        y_avg = g.values

        if filter_avg and len(y_avg) > 0:
            # optional spike suppression (same length)
            y_avg = self.despike_median(y_avg, window=8)

        # ---- Figure (reuse or create) ----
        created_here = False
        if fig is None:
            fig = go.Figure()
            created_here = True

        # Raw samples
        if not avg_only:
            fig.add_trace(go.Scattergl(
                x=dt,
                y=y_all,
                mode="markers",
                marker=dict(size=4, opacity=0.2),
                name="Samples",
                showlegend=False,
            ))

        # Average line
        fig.add_trace(go.Scatter(
            x=x_avg,
            y=y_avg,
            mode="lines",
            line=dict(width=2 if avg_only else 1, color="black" if not avg_only else None),
            name=label,
        ))

        # Choose y-axis ticks
        max_y = max(y_avg)
        y_ticks = np.logspace(np.log10(10), np.log10(max_y * 1.5), 12)
        y_ticks = [int(np.floor(y / 10) * 10) for y in y_ticks]


        # ---- Axes / layout ----
        fig.update_layout(
            width=1000,
            height=400,
            title=title,
            template="plotly_white",
            yaxis=dict(
                title="Price per SQFT ($)",
                type="log",
                range=[np.log10(50), np.log10(2000)],  # matches [50, 2000]
                tickmode="array",
                tickvals=y_ticks,
                ticktext=[str(v) for v in y_ticks],
                showgrid=True,
            ),
            xaxis=dict(
                title="Time",
                type="date",
                tickformat="%m/%Y",       # month/year labels
                dtick="M24",              # every 24 months = every 2 years
                tickangle=45,             # tilt labels 45 deg
                showgrid=True,
            ),
            margin=dict(l=60, r=20, t=60, b=80),
        )

        # If we created the fig here and the caller didn’t pass one, show it (only for full plot)
        if created_here and not avg_only:
            fig.show()

        return fig

    def despike_median(self, x, window=5):
        """
        Replace x with median-filtered version.
        window must be odd; bigger -> more smoothing (and more spike suppression).
        """
        if window % 2 == 0:
            window += 1
        pad = window // 2
        # reflect padding at both ends
        x_padded = np.pad(x, pad, mode="reflect")
        # sliding median
        return np.array([
            np.median(x_padded[i:i+window]) 
            for i in range(len(x))
        ])


    # @classmethod
    # def retrun_all_cities(cls, dp_path):
    #     """Return a list of all unique cities in the dataset"""

    #     return self.data_raw['city'].unique(


class RentCastPredictor():

    DEFAULT_NUM_COLS = ['bedrooms', 'bathrooms', 'months', 'log_sqft', 'log_lotSize', 'yearsOld', 'log_cluster_time_median']
    DEFAULT_CAT_COLS = ["city", "geo_cluster", 'sale_month']
    # DEFAULT_CAT_COLS = ["city"]
    DEFAULT_TARGET_COL = 'log_lastSalePrice'

    def __init__(self, df_raw, num_columns=None, cat_columns=None, target_col=None, random_state=None):
        df_raw = copy.deepcopy(df_raw)
        df_raw.reset_index(drop=True, inplace=True)

        self.random_state = np.random.randint(0, 100) if random_state is None else random_state

        # Model columns
        self.num_cols = self.DEFAULT_NUM_COLS if num_columns is None else num_columns
        self.cat_cols = self.DEFAULT_CAT_COLS if cat_columns is None else cat_columns
        self.target_col = self.DEFAULT_TARGET_COL if target_col is None else target_col
        self.derived_cols = []

        # Data
        self.df_raw = df_raw
        self.df_filtered = None
        self.df_derived = None
        self.df_normalized = None
        self.df_clean = None
        self.df_predict = None
        self.model = None


        # Pre-process the data
        self.filter_properties()
        self.derive_features()
        self.normalize_data()

    @property
    def feature_cols(self):
        return self.num_cols + self.cat_cols + [self.target_col]

    def filter_properties(self):
        """
        Filter out properties that are not in the list of cities.
        """
        df = copy.deepcopy(self.df_raw)

        # Drop rows missing data needed for prediction
        feature_cols = [self.target_col] + self.num_cols + self.cat_cols
        for col in feature_cols:
            if col in list(df):
                df = df.dropna(subset=col)

        # Filter out properties that are not in the list of cities.
        df = df[
            (df["lastSalePrice"] < 2_000_000) &
            (df["bedrooms"].between(1, 8)) &
            (df["bathrooms"].between(1, 8)) &
            (df["squareFootage"].between(500, 6000)) &
            (df["lotSize"].between(500, 25_000)) &
            (df["yearBuilt"].between(1850, 2025)) &
            (df["lastSalePrice"] / df["squareFootage"] < 2000) & 
            (df["lastSalePrice"] / df["squareFootage"] > 30)
        ]

        self.df_filtered = df


    def derive_features(self):
        """
        Derive features from the raw data.
        """
        df = copy.deepcopy(self.df_filtered)
        
        # Derive sale month and sale year and ensure they are integers
        df[["sale_month" ,"sale_year"]] = df["month-year"].str.split("-", expand=True)
        df['sale_year'] = df['sale_year'].astype(int)
        df["sale_month"] = df["sale_month"].astype(int)
        # self.num_cols += ['sale_month', 'sale_year']

        # Derive geo cluster
        coords = df[["latitude","longitude"]]
        df["geo_cluster"] = KMeans(n_clusters=50, random_state=42).fit_predict(coords)
        cluster_means = df.groupby("geo_cluster")["lastSalePrice"].median()
        # self.cat_cols += ['geo_cluster']

        # Log last sale price
        df['log_lastSalePrice'] = np.log1p(df['lastSalePrice'].values) 

        # Median price of the geo_cluster per month
        for cluster_id, group in df.groupby("geo_cluster"):
            X_temp = group[["months"]].values
            y_temp = group["log_lastSalePrice"].values

            # simple linear fit
            temp_model = LinearRegression()
            temp_model.fit(X_temp, y_temp)

            # predict for this group's rows
            df.loc[group.index, "log_cluster_time_median"] = temp_model.predict(X_temp)

        # self.num_cols += ['log_cluster_time_median']
        self.df_derived = df

    def normalize_data(self):

        df = copy.deepcopy(self.df_derived)
    
        df['log_lotSize'] = np.log(df['lotSize'].values)

        df['log_sqft'] = np.log(df['squareFootage'].values)

        df['lot_sqft_per_sqft'] = np.log(df['lotSize'].values / df['squareFootage'].values)

        df['log_sqFt_per_bedroom'] = np.log(df['squareFootage'].values) / df['bedrooms'].values

        df['log_sqFt_per_bathroom'] = np.log(df['squareFootage'].values) / df['bathrooms'].values

        df['yearsOld'] = 2025 - df['yearBuilt'].values

        self.df_normalized = df

    def setup_model(self):
        """
        Setup the model.
        """
        num_transformer = StandardScaler()
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers = [
            ("num", num_transformer, self.num_cols),
            ("cat", cat_transformer, self.cat_cols)
        ]

        preprocessor = ColumnTransformer(transformers)
    
        hgb = HistGradientBoostingRegressor(
            max_iter=400, 
            learning_rate=0.01, 
            max_depth=10,
            early_stopping=True,
            random_state=self.random_state
        )

        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", hgb)
        ])

        # Chose dataset
        if self.df_normalized is not None:
            self.df_clean = self.df_normalized
        elif self.df_derived is not None:
            self.df_clean = self.df_derived
        elif self.df_filtered is not None:
            self.df_clean = self.df_filtered
        else:
            self.df_clean = self.df_raw

        self.model = model

    def train_model_and_eval(self):

        # Input and output vars
        X = self.df_clean[self.num_cols + self.cat_cols]
        y = self.df_clean[self.target_col]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        # Fit
        self.model.fit(X_train, y_train)

        # Cross-validation score
        rmse = cross_val_score(self.model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
        rmse = -np.mean(rmse)
        rsq = self.model.score(X_test, y_test) * 100  # R^2
        print("CV RMSE:", round(rmse, 2), ", Test R²:", round(rsq, 1))

        return rmse, rsq

    def check_one(self, n=1):

        # pick a row by index (from the same schema as training)
        i = np.random.randint(0, self.df_clean.shape[0], n)
        idx = self.df_clean.index[i]
        df_predict = self.predict(self.df_clean.loc[idx, :], plot=False)

        pred_price_k = df_predict.loc[idx, 'pred_price_k'].values[0]
        act_price_k = df_predict.loc[idx, 'act_price_k'].values[0]
        error_k = df_predict.loc[idx, 'error_k'].values[0]
        per_error = df_predict.loc[idx, 'per_error'].values[0]
        
        if n == 1:
            print("Predicted - Actual - Error - %% Error --- $%i - $%i - %i - %.1f%%" % (pred_price_k, act_price_k, error_k, per_error))
        return df_predict.loc[idx, ['city', 'month-year', 'bedrooms', 'bathrooms', 'squareFootage', 'yearBuilt', 'pred_price_k', 'act_price_k', 'error_k', 'per_error']]


    def predict(self, df=None, plot=False):
        """
        Predict the price of a property.
        """

        if df is None:
            df = copy.deepcopy(self.df_clean)

        feature_cols = self.num_cols + self.cat_cols + [self.target_col]
        x_one = df.loc[:, feature_cols]

        pred_log = self.model.predict(x_one)
        pred_price = np.expm1(pred_log).astype(float)

        # Manual empiracl linear transformation
        # pred_price = (pred_price - 0) / (1 - 800e3 / 2000e3) 
        # pred_price = (pred_price) * 1.2 - 80e3

        act_price = df.loc[:, 'lastSalePrice'].values
        error = pred_price - act_price
        per_error = (pred_price - act_price) / act_price * 100

        pred_price_k = np.round(pred_price / 1e3)
        act_price_k = np.round(act_price / 1e3)
        error_k = np.round(error / 1e3)
        per_error = np.round(per_error, 1)
        
        df.loc[:, 'pred_price_k'] = pred_price_k
        df.loc[:, 'act_price_k'] = act_price_k
        df.loc[:, 'error_k'] = error_k
        df.loc[:, 'per_error'] = per_error

        if plot:

            n = pred_price_k.shape[0]
            # alpha = (35510449 - 49 * n) / 35510400

            plt.plot(np.array(pred_price_k), np.array(act_price_k), lw=0, ms=4, mec='black', alpha=0.002, marker='o')
            plt.grid(True)

            ax = plt.gca()
            # get limits
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]

            # plot 1:1 line
            ax.plot(lims, lims, 'k--', alpha=1)  # black dashed line
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            plt.ylabel('Actual Cost [$1k]')
            plt.xlabel('Predicted Cost [$1k]')
            plt.title('Predicted vs Actual Cost')

        self.df_predict = df
        return self.df_predict

    def plot_residuals(self, per=False):

        df_predict = self.df_predict
        error = df_predict['per_error'].values if per else df_predict['error_k'].values
            
        important_cols = []
        # important_cols = ['months', 'squareFootage', 'sale_year', 'yearsOld', 'lastSalePrice', 'geo_cluster']
        important_cols += self.feature_cols
        important_cols += ['lastSalePrice']

        plt.figure(figsize=(12, 12))

        for idx, important_col in enumerate(important_cols):

            col_data = df_predict.loc[:, important_col].values

            if important_col == 'lastSalePrice':
                col_data = col_data / 1e3

            plt.subplot(4, 3, idx+1)
            plt.plot(col_data, error, lw=0, ms=4, marker='o', mec='black', alpha=0.002)
            plt.xlabel(important_col)
            plt.grid()
            if per:
                plt.title('Error [%%] vs %s' % important_col)
                plt.ylim([-200, 200])
            else:
                plt.title('Error [$1k] vs %s' % important_col)

        plt.tight_layout()

    def plot_binned_residuals(self, bins=20, strategy="quantile"):
        """
        Plots mean residual per bin with a shaded 10–90% band.
        """
        
        df = self.df_predict

        plt.figure(figsize=(12, 12))

        important_cols = []
        important_cols += self.num_cols
        important_cols += ['lastSalePrice']

        # Loop over all features
        for idx, col in enumerate(important_cols):

            # Get stats
            error = df['error_k'].values
            col_data = df[col].values
            stats = self.binned_residual_stats(col_data, error, bins=bins, strategy=strategy)

            x_mid = stats["x_mid"]
            err_mean = stats["err_mean"]
            err_p10 = stats["err_p10"]
            err_p90 = stats["err_p90"]

            # Plot
            plt.subplot(4, 3, idx+1)

            plt.axhline(0, ls="--", lw=1, color="gray")
            plt.fill_between(x_mid, err_p10, err_p90, alpha=0.2, edgecolor="none")
            plt.plot(x_mid, err_mean, marker="o", lw=2, mec='black')
            plt.xlabel(col)
            plt.ylabel("Mean error")
            plt.title(col)
            plt.ylim([-300, 300])
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

    def binned_residual_stats(self, x, err, bins=20, strategy="quantile"):
        """
        x: 1D array-like feature
        err: 1D array-like residuals (e.g., (pred - actual)/actual*100 or pred-actual)
        bins: number of bins
        strategy: "quantile" (equal-count bins) or "uniform" (equal-width bins)
        Returns: DataFrame with bin centers, mean, p10, p90, count
        """
        x = pd.Series(x).astype(float)
        err = pd.Series(err).astype(float)

        if strategy == "quantile":
            # quantile bins handle skewed features better
            q = np.linspace(0, 1, bins + 1)
            edges = x.quantile(q).values
            # ensure strictly increasing (dedup if constant segments)
            edges = np.unique(edges)
            if len(edges) < 3:  # too few unique edges
                edges = np.linspace(x.min(), x.max(), min(bins, 10) + 1)
            cats = pd.cut(x, bins=edges, include_lowest=True)
        else:
            cats = pd.cut(x, bins=bins)

        g = pd.DataFrame({"x": x, "err": err, "bin": cats}).dropna().groupby("bin", observed=False)
        stats = g.agg(
            x_mid=("x", lambda s: (s.min() + s.max()) / 2.0),
            err_mean=("err", "mean"),
            err_p10=("err", lambda s: np.percentile(s, 10)),
            err_p90=("err", lambda s: np.percentile(s, 90)),
            count=("err", "size"),
        ).reset_index(drop=True)

        return stats.sort_values("x_mid")