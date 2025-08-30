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
        print([t[0] for t in tables])  # list of table names
                
    def read_city(self, city, state):

        # Read data from the specified table
        """
        Read the data for a given city and state from the SQLite database.

        Parameters:
            city (str): The city (e.g. "San Francisco").
            state (str): The state (e.g. "CA").

        Returns:
            None
        """
        table_name = self._return_table_name(city, state)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)

        # Get data and process it
        self.data_raw[table_name] = df
        if table_name not in list(self.data_processed):
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
    
    def plot_city_states(self, city_states):
        fig = go.Figure()
        for (city, state) in city_states:
            self.plot_city(city, state, avg_only=True, fig=fig)
        # plt.legend(loc='best')
        fig.show()
    
    def plot_city(self, city, state, avg_only=False, fig=None):

        self.read_city(city, state)
        df = self.data_processed[self._return_table_name(city, state)]

        self.plot_processed_trace_ptly(df, avg_only=avg_only, fig=fig)

        # plt.title('Price per SQFT - %s, %s' % (city, state))

    def plot_processed_trace_mpl(self, df, filter_avg=True, avg_only=False):

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

        # ---- Axes / layout ----
        y_ticks = [10, 30, 100, 300, 1000]
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