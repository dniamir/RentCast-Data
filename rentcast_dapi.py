"""Rent Cast API Python Helper Classes"""
import pandas as pd
import numpy as np
import datetime
import copy
import requests
import json


class RentCastData():

    URL = 'https://api.rentcast.io/v1/properties'
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
                 api_key=''):
        
        # Initalizations
        self.headers = {'accept': 'application/json', 'X-Api-Key': api_key}
        self.querystring = {}
        self.data_raw = None
        self.data_processed = None

        # Add input args to address string
        if address is not None:
            self.querystring['address'] = address
        if city is not None:
            self.querystring['city'] = city
        if state is not None:
            self.querystring['state'] = state
        if zip is not None:
            self.querystring['zip'] = zip
        if limit is not None:
            self.querystring['limit'] = limit

        # Loop through offsets
        querystring = copy.deepcopy(self.querystring)
        offsets = np.arange(0, offset_lim, 1)
        for offset in offsets:
            querystring['offset'] = limit * offset

            # Get query
            response = requests.get(self.URL, 
                                    headers=self.headers, 
                                    params=querystring)
            listings = response.json()

            # Check that a response was given
            if not len(listings):
                print("No more listings at offset %i for limit %i" % (offset, limit))
                break

            df_response = self.parse_response(listings)

            # Add query properties to response
            df_response['Offset'] = limit * offset
            df_response['limit'] = limit
            df_response['Querystring'] = str(querystring)

            # Add data to existing data
            self.data_raw = pd.concat([self.data_raw, df_response], ignore_index=True)

        # Add processed metrics
        self.add_metrics()

    def parse_response(self, listings):

        df_all = None

        # Loop through results and create dataframe
        for listing_loop in listings:

            listing = copy.deepcopy(listing_loop)
            listing_temp = copy.deepcopy(listing)

            for header in listing_temp:
                if header not in self.HEADERS:
                    del listing[header]
                else:
                    listing[header] = [listing[header]]

            df = pd.DataFrame.from_dict(listing)
            df_all = pd.concat([df_all, df], ignore_index=True)

        return df_all

    def add_months(self, date, months):
        # Calculate the target month and year
        target_month = date.month + months
        target_year = date.year + (target_month - 1) // 12
        target_month = (target_month - 1) % 12 + 1

        # Calculate the day
        day = min(date.day, [31,
                            29 if target_year % 4 == 0 and (target_year % 100 != 0 or target_year % 400 == 0) else 28,
                            31, 30, 31, 30, 31, 31, 30, 31, 30, 31][target_month - 1])

        return date.replace(year=target_year, month=target_month, day=day)

    def __months_between(self, date1, date2):
        # Ensure date1 is earlier than date2
        if date1 > date2:
            date1, date2 = date2, date1
        
        # Calculate the difference in years and months
        year_diff = date2.year - date1.year
        month_diff = date2.month - date1.month
        
        # Total number of months
        total_months = year_diff * 12 + month_diff
        
        return total_months

    def add_metrics(self):
        """Add metrics of interest to the dataset"""

        df = copy.deepcopy(self.data_raw)
        df = df.dropna(subset=['lastSalePrice', 'squareFootage', 'lastSaleDate'])

        # Price per Square Foot
        df['price_per_sqft'] =  df['lastSalePrice'] / df['squareFootage']

        # Duration on market
        df = self.parse_sale_date(df)

        self.data_processed = df

    def parse_sale_date(self, df):
        """Parse last sale date into more readable terms"""

        # Get duration in months
        df.sort_values('lastSaleDate', inplace=True)
        for index, row in df.iterrows():
            dt = datetime.datetime.strptime(row['lastSaleDate'], '%Y-%m-%dT%H:%M:%S.%fZ')
            df.loc[index, 'month'] = dt.strftime('%m')
            df.loc[index, 'year'] = dt.strftime('%Y')
            df.loc[index, 'datetime'] = dt
        
        return df

    def parse_sale_date2(self, df):
        """Parse last sale date into more readable terms"""

        # Get duration in months
        df.sort_values('lastSaleDate', inplace=True)
        t0 = datetime.datetime.strptime(df['lastSaleDate'].values[0], '%Y-%m-%dT%H:%M:%S.%fZ')
        for index, row in df.iterrows():
            dt = datetime.datetime.strptime(row['lastSaleDate'], '%Y-%m-%dT%H:%M:%S.%fZ')
            df.loc[index, 'months'] = self.__months_between(dt, t0)
            df.loc[index, 'month-year'] = dt.strftime('%m-%Y')
            df.loc[index, 'datetime'] = dt

        return df