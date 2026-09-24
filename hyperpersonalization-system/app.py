import os   
import json
import re
import uuid
import time
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flask_session import Session
import logging
from logging.handlers import RotatingFileHandler
from qdrant_client.http import models
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from google.oauth2 import service_account
from pymongo import MongoClient
import numpy as np
import vertexai
from bson import ObjectId
import random
from collections import defaultdict
import hashlib
from langfuse import Langfuse
from opik.integrations.langchain import OpikTracer
import dateutil.parser
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import threading
import ssl

# Load environment variables
load_dotenv()

# Enhanced logging setup
import sys

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create dedicated loggers
demographic_logger = logging.getLogger('demographic_recommendations')
other_users_logger = logging.getLogger('other_users_recommendations')
interactions_logger = logging.getLogger('user_interactions')
preferences_logger = logging.getLogger('user_preferences')
birthday_logger = logging.getLogger('birthday_celebrations')
email_logger = logging.getLogger('email_triggers') # NEW: Email logger

# Set up file handlers
if not os.path.exists('logs'):
    os.makedirs('logs')

def create_utf8_handler(filename):
    handler = RotatingFileHandler(filename, maxBytes=30000000, backupCount=5, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    return handler

demographic_logger.addHandler(create_utf8_handler('logs/demographic_recommendations.log'))
demographic_logger.setLevel(logging.INFO)

other_users_logger.addHandler(create_utf8_handler('logs/other_users_recommendations.log'))
other_users_logger.setLevel(logging.INFO)

interactions_logger.addHandler(create_utf8_handler('logs/user_interactions.log'))
interactions_logger.setLevel(logging.INFO)

preferences_logger.addHandler(create_utf8_handler('logs/user_preferences.log'))
preferences_logger.setLevel(logging.INFO)

birthday_logger.addHandler(create_utf8_handler('logs/birthday_celebrations.log'))
birthday_logger.setLevel(logging.INFO)

email_logger.addHandler(create_utf8_handler('logs/email_triggers.log')) # NEW: Email logger handler
email_logger.setLevel(logging.INFO)

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(mongo_uri)
db = mongo_client["personalization_database"]

# Collections
products_collection = db["products"]
demographics_collection = db["demographics"]
user_preferences_collection = db["user_preferences"]  
cart_collection = db["cart"]
users_collection = db["users_collection"]
user_interactions_collection = db["user_interactions"]
events_collection = db["events"]
email_triggers_collection = db["email_triggers"] # NEW: Email Triggers collection

# Qdrant connection
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if qdrant_api_key:
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
else:
    qdrant_client = QdrantClient(url=qdrant_url)
# Flask app setup
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'change_me_in_env')

# Global variables for AI components
embeddings_model = None
llm_model = None
current_search_results = {}
last_search_results = {}
opik_tracer = OpikTracer()

# =================== NEW: EMAIL SENDING UTILITY ===================

def send_email(to_email, subject, html_content):
    """Sends an HTML email using SMTP credentials from .env."""
    sender_email = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))

    if not all([sender_email, password, smtp_server]):
        email_logger.error("SMTP credentials are not configured in .env file.")
        return False

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = f"InsightRecs Recommendations <{sender_email}>"
    message["To"] = to_email
    message.attach(MIMEText(html_content, "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(sender_email, password)
            server.sendmail(sender_email, to_email, message.as_string())
        email_logger.info(f"Email sent successfully to {to_email} with subject: '{subject}'")
        return True
    except Exception as e:
        email_logger.error(f"Failed to send email to {to_email}: {e}")
        return False

# =================== NEW: HTML EMAIL TEMPLATES ===================
def get_cart_email_template(user_name, cart_items, total_price):
    """Generates the HTML content for the cart reminder email."""
    items_html = ""
    for item in cart_items:
        image_url = item.get('image_url') or 'https://via.placeholder.com/60'
        # In the previous version, image_url was used but the get_direct_image_url was not called
        # This is corrected here to ensure images are displayed
        if 'http' not in image_url:
            image_url = get_direct_image_url(image_url)

        items_html += f"""
        <tr>
            <td style="padding: 15px 10px; border-bottom: 1px solid #dee2e6;">
                <img src="{image_url}" alt="{item['name']}" width="60" style="border-radius: 5px;">
            </td>
            <td style="padding: 15px 10px; border-bottom: 1px solid #dee2e6; color: #495057;">
                {item['name']}
            </td>
            <td style="padding: 15px 10px; border-bottom: 1px solid #dee2e6; color: #495057; text-align: center;">
                {item['quantity']}
            </td>
            <td style="padding: 15px 10px; border-bottom: 1px solid #dee2e6; color: #495057; text-align: right;">
                {item['price']}
            </td>
        </tr>
        """
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>You left items in your cart!</title>
    </head>
    <body style="font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f1f3f6;">
        <table width="100%" border="0" cellspacing="0" cellpadding="0" style="background-color: #f1f3f6;">
            <tr>
                <td align="center">
                    <table width="600" border="0" cellspacing="0" cellpadding="20" style="max-width: 600px; margin: 20px auto; background-color: #ffffff; border-radius: 8px;">
                        <tr>
                            <td align="center" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; padding: 20px;">
                                <h1 style="color: #ffffff; margin: 0; font-size: 28px;">InsightRecs</h1>
                                <p style="color: #ffffff; margin: 5px 0 0 0; opacity: 0.9;">AI Powered Hyper-Personalized Recommendations</p>
                            </td>
                        </tr>

                        <tr>
                            <td style="padding: 30px 20px;">
                                <h2 style="font-size: 22px; color: #333333; margin-top: 0;">Hi {user_name},</h2>
                                <p style="color: #555555; font-size: 16px; line-height: 1.6;">
                                    It looks like you left some items in your shopping cart. Don't miss out, complete your purchase today!
                                </p>
                                
                                <table width="100%" border="0" cellspacing="0" cellpadding="0" style="margin-top: 20px; border-collapse: collapse;">
                                    <thead>
                                        <tr>
                                            <th colspan="2" style="padding: 10px; border-bottom: 2px solid #adb5bd; color: #343a40; text-align: left;">Product</th>
                                            <th style="padding: 10px; border-bottom: 2px solid #adb5bd; color: #343a40; text-align: center;">Quantity</th>
                                            <th style="padding: 10px; border-bottom: 2px solid #adb5bd; color: #343a40; text-align: right;">Price</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {items_html}
                                        <tr>
                                            <td colspan="3" style="padding: 15px 10px; text-align: right; font-weight: bold; color: #343a40;">Total:</td>
                                            <td style="padding: 15px 10px; text-align: right; font-weight: bold; color: #28a745;">{total_price}</td>
                                        </tr>
                                    </tbody>
                                </table>

                                <table width="100%" border="0" cellspacing="0" cellpadding="0" style="margin-top: 30px;">
                                    <tr>
                                        <td align="center">
                                            <a href="http://your-website-url.com/cart" target="_blank" style="background: linear-gradient(135deg, #28a745 0%, #218838 100%); color: #ffffff; padding: 12px 25px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">
                                                Complete Your Purchase
                                            </a>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>

                        <tr>
                            <td align="center" style="padding: 20px; font-size: 12px; color: #999999;">
                                <p style="margin: 0;">This is a reminder about the items in your cart at InsightRecs.</p>
                                <p style="margin: 5px 0 0 0;">&copy; {datetime.now().year} InsightRecs. All rights reserved.</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

def get_events_email_template(user_name, events):
    """Generates the HTML content for the events and offers email."""
    events_html = ""
    for event in events:
        offers_html = ""
        # Limit to 3 offers per event for a clean email
        for offer in event.get('offers', [])[:3]:
            product = offer.get('product', {})
            image_url = product.get('image', 'https://via.placeholder.com/80')

            offers_html += f"""
            <div style="background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; margin-bottom: 15px; display: table; width: 100%;">
                <div style="display: table-cell; vertical-align: top; width: 100px;">
                    <img src="{image_url}" alt="{product.get('name', 'Offer')}" width="80" style="border-radius: 5px;">
                </div>
                <div style="display: table-cell; vertical-align: top;">
                    <h4 style="margin: 0; font-size: 16px; color: #343a40;">{product.get('name', 'Special Offer')}</h4>
                    <p style="margin: 5px 0; color: #d9534f; font-weight: bold; font-size: 15px;">
                        {offer.get('discount_percentage', '')}% OFF - Now {offer.get('discounted_price', '')}
                    </p>
                    <p style="font-size: 13px; color: #6c757d; margin: 5px 0 0 0;">
                        <i>{offer.get('offer_description', '')}</i>
                    </p>
                </div>
            </div>
            """
        
        events_html += f"""
        <h3 style="color: #007bff; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px; margin-top: 30px;">
            {event['event_name']}
        </h3>
        <p style="color: #555555; font-size: 16px; line-height: 1.6;">{event['event_description']}</p>
        {offers_html}
        """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Latest Offers from InsightRecs</title>
    </head>
    <body style="font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f1f3f6;">
        <table width="100%" border="0" cellspacing="0" cellpadding="0" style="background-color: #f1f3f6;">
            <tr>
                <td align="center">
                    <table width="600" border="0" cellspacing="0" cellpadding="20" style="max-width: 600px; margin: 20px auto; background-color: #ffffff; border-radius: 8px;">
                        <tr>
                            <td align="center" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; padding: 20px;">
                                <h1 style="color: #ffffff; margin: 0; font-size: 28px;">InsightRecs</h1>
                                <p style="color: #ffffff; margin: 5px 0 0 0; opacity: 0.9;">AI Powered Hyper-Personalized Recommendations</p>
                            </td>
                        </tr>

                        <tr>
                            <td style="padding: 30px 20px;">
                                <h2 style="font-size: 22px; color: #333333; margin-top: 0;">Hi {user_name},</h2>
                                <p style="color: #555555; font-size: 16px; line-height: 1.6;">
                                    Check out the latest deals and offers available now!
                                </p>
                                {events_html}
                                <table width="100%" border="0" cellspacing="0" cellpadding="0" style="margin-top: 30px;">
                                    <tr>
                                        <td align="center">
                                            <a href="http://your-website-url.com/dashboard" target="_blank" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff; padding: 12px 25px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">
                                                Explore All Offers
                                            </a>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>

                        <tr>
                            <td align="center" style="padding: 20px; font-size: 12px; color: #999999;">
                                <p style="margin: 0;">You received this email as a registered user of InsightRecs.</p>
                                <p style="margin: 5px 0 0 0;">&copy; {datetime.now().year} InsightRecs. All rights reserved.</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

def get_recommendations_email_template(user_name, products):
    """Generates the HTML content for product recommendations."""
    products_html = ""
    for product in products:
        # Each product is a row containing a styled card
        products_html += f"""
        <tr>
            <td style="padding: 10px 0;">
                <div style="background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; display: table; width: 100%;">
                    <div style="display: table-cell; vertical-align: top; width: 100px;">
                        <img src="{product.get('image', 'https://via.placeholder.com/80')}" alt="{product['name']}" width="80" style="border-radius: 5px;">
                    </div>
                    <div style="display: table-cell; vertical-align: top;">
                        <h4 style="margin: 0; font-size: 16px; color: #343a40;">{product['name']}</h4>
                        <p style="margin: 5px 0; color: #28a745; font-weight: bold; font-size: 15px;">{product['price']}</p>
                        {f'<p style="font-size: 13px; color: #6c757d; margin: 5px 0 0 0;"><i>{product["reason"]}</i></p>' if product.get("reason") else ""}
                    </div>
                </div>
            </td>
        </tr>
        """

    # This is the full email body that wraps the products
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Your InsightRecs Recommendations</title>
    </head>
    <body style="font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f1f3f6;">
        <table width="100%" border="0" cellspacing="0" cellpadding="0" style="background-color: #f1f3f6;">
            <tr>
                <td align="center">
                    <table width="600" border="0" cellspacing="0" cellpadding="20" style="max-width: 600px; margin: 20px auto; background-color: #ffffff; border-radius: 8px;">
                        <tr>
                            <td align="center" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; padding: 20px;">
                                <h1 style="color: #ffffff; margin: 0; font-size: 28px;">InsightRecs</h1>
                                <p style="color: #ffffff; margin: 5px 0 0 0; opacity: 0.9;">AI Powered Hyper-Personalized Recommendations</p>
                            </td>
                        </tr>

                        <tr>
                            <td style="padding: 30px 20px;">
                                <h2 style="font-size: 22px; color: #333333; margin-top: 0;">Hi {user_name}!</h2>
                                <p style="color: #555555; font-size: 16px; line-height: 1.6;">
                                    Based on your recent activity, we've found some products we think you'll love.
                                </p>
                                
                                <h3 style="color: #007bff; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px; margin-top: 30px;">
                                    Featured For You
                                </h3>

                                <table width="100%" border="0" cellspacing="0" cellpadding="0">
                                    {products_html}
                                </table>

                                <table width="100%" border="0" cellspacing="0" cellpadding="0" style="margin-top: 30px;">
                                    <tr>
                                        <td align="center">
                                            <a href="http://localhost:5000/dashboard" target="_blank" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff; padding: 12px 25px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">
                                                Discover More
                                            </a>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>

                        <tr>
                            <td align="center" style="padding: 20px; font-size: 12px; color: #999999;">
                                <p style="margin: 0;">You received this email because you requested recommendations from InsightRecs.</p>
                                <p style="margin: 5px 0 0 0;">&copy; 2025 InsightRecs. All rights reserved.</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

def get_birthday_email_template(user_name, birthday_data):
    """Generates the HTML content for the birthday email."""
    first_name = user_name.split()[0] if user_name else "User"
    offers_html = ""
    for offer in birthday_data.get('offers', []):
        product = offer.get('product', {})
        image_url = product.get('image', 'https://via.placeholder.com/80')
        offers_html += f"""
        <div style="background-color: #fff9e6; border: 1px solid #ffecb3; border-left: 4px solid #ffca28; border-radius: 8px; padding: 15px; margin-bottom: 15px; display: table; width: 100%;">
            <div style="display: table-cell; vertical-align: top; width: 100px;">
                <img src="{image_url}" alt="{product.get('name', 'Birthday Gift')}" width="80" style="border-radius: 5px;">
            </div>
            <div style="display: table-cell; vertical-align: top;">
                <h4 style="margin: 0; font-size: 16px; color: #343a40;">{product.get('name', 'Special Offer')}</h4>
                <p style="margin: 5px 0; color: #d9534f; font-weight: bold; font-size: 15px;">
                    {offer.get('discount_percentage', '')}% OFF - Now {offer.get('discounted_price', '')}
                </p>
                <p style="font-size: 13px; color: #6c757d; margin: 5px 0 0 0;">
                    <i>{offer.get('offer_description', '')}</i>
                </p>
            </div>
        </div>
        """
        
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Happy Birthday, {first_name}!</title>
    </head>
    <body style="font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f1f3f6;">
        <table width="100%" border="0" cellspacing="0" cellpadding="0" style="background-color: #f1f3f6;">
            <tr>
                <td align="center">
                    <table width="600" border="0" cellspacing="0" cellpadding="20" style="max-width: 600px; margin: 20px auto; background-color: #ffffff; border-radius: 8px;">
                        <tr>
                            <td align="center" style="background: linear-gradient(135deg, #ff8c00 0%, #ffc107 100%); border-radius: 8px; padding: 20px;">
                                <h1 style="color: #ffffff; margin: 0; font-size: 28px;">🎂 Happy Birthday! 🎂</h1>
                            </td>
                        </tr>

                        <tr>
                            <td style="padding: 30px 20px;">
                                <h2 style="font-size: 22px; color: #333333; margin-top: 0;">{birthday_data.get('celebration_message', f'Happy Birthday, {user_name}!')}</h2>
                                <p style="color: #555555; font-size: 16px; line-height: 1.6;">
                                    {birthday_data.get('celebration_subtitle', 'We wish you a fantastic day! To celebrate, we have prepared some exclusive offers just for you.')}
                                </p>
                                
                                <h3 style="color: #c0392b; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px; margin-top: 30px;">
                                    Your Birthday Presents 🎁
                                </h3>
                                {offers_html}
                                <table width="100%" border="0" cellspacing="0" cellpadding="0" style="margin-top: 30px;">
                                    <tr>
                                        <td align="center">
                                            <a href="http://your-website-url.com/dashboard" target="_blank" style="background: linear-gradient(135deg, #ff8c00 0%, #ffc107 100%); color: #ffffff; padding: 12px 25px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">
                                                Claim Your Gifts
                                            </a>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>

                        <tr>
                            <td align="center" style="padding: 20px; font-size: 12px; color: #999999;">
                                <p style="margin: 0;">These offers are exclusive to you for your birthday celebration.</p>
                                <p style="margin: 5px 0 0 0;">&copy; {datetime.now().year} InsightRecs. All rights reserved.</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """
# =================== NEW: EMAIL TRIGGER LOGIC ===================

def send_cart_email(user_id):
    with app.app_context():
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return
        
        cart_items = get_user_cart(user_id)
        if not cart_items:
            email_logger.info(f"Cart is empty for user {user_id}. Skipping cart reminder email.")
            return

        total_price = sum(float(re.search(r'[\d.]+', item.get('price', '$0')).group()) * item.get('quantity', 1) for item in cart_items)
        html_content = get_cart_email_template(user['name'], cart_items, f"${total_price:.2f}")
        send_email(user['email'], "You left something in your cart!", html_content)
        email_triggers_collection.update_one(
            {'user_id': user_id, 'email_type': 'cart', 'status': 'scheduled'},
            {'$set': {'status': 'sent', 'sent_at': datetime.now()}}
        )

def send_events_email(user_id):
    with app.app_context():
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return
            
        events = get_current_events(user_id)
        if not events:
            email_logger.info(f"No active events. Skipping offers email for user {user_id}.")
            return

        html_content = get_events_email_template(user['name'], events)
        send_email(user['email'], "Latest Offers & Events Just For You!", html_content)
        email_triggers_collection.update_one(
            {'user_id': user_id, 'email_type': 'events', 'status': 'scheduled'},
            {'$set': {'status': 'sent', 'sent_at': datetime.now()}}
        )


def send_recommendations_email(user_id, products):
    """Send only the first section products via email"""
    with app.app_context():
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            email_logger.error(f"User not found for sending recommendations: {user_id}")
            return

        if not products:
            email_logger.info(f"No products to recommend for user {user_id}. Skipping email.")
            return

        # Convert products to the format expected by email template
        formatted_products = []
        for product in products:
            if isinstance(product, str):
                # If product is just an ID, fetch from database
                try:
                    product_obj = products_collection.find_one({'_id': ObjectId(product)})
                    if product_obj:
                        formatted_products.append(format_product_for_email(product_obj))
                except:
                    continue
            elif isinstance(product, dict):
                # If product is already a dict, format it
                formatted_products.append(format_product_for_email(product))

        if not formatted_products:
            email_logger.info(f"No valid products to send for user {user_id}")
            return

        # Send email with subject indicating it's from the personalized section
        html_content = get_recommendations_email_template(user['name'], formatted_products)
        send_email(user['email'], "Your Personalized Shopping Trends - Recommendations!", html_content)
        email_logger.info(f"Sent {len(formatted_products)} personalized products (first section) to {user['email']}")

def format_product_for_email(product):
    """Format product data for email template"""
    try:
        # Get image URL
        image_url = product.get('image') or product.get('image_url') or 'https://via.placeholder.com/80'
        if image_url and 'http' not in image_url:
            image_url = get_direct_image_url(image_url) or 'https://via.placeholder.com/80'
        
        return {
            'name': product.get('name', 'Unknown Product'),
            'price': product.get('price', '$0.00'),
            'image': image_url,
            'reason': product.get('reason', ''),
            'brand': product.get('brand', ''),
            'category': product.get('category', ''),
            'rating': product.get('rating', 4.0),
            '_id': str(product.get('_id', ''))
        }
    except Exception as e:
        email_logger.error(f"Error formatting product for email: {e}")
        return {
            'name': 'Product',
            'price': '$0.00', 
            'image': 'https://via.placeholder.com/80',
            'reason': '',
            'brand': '',
            'category': ''
        }


def send_birthday_email(user_id, birthday_data):
    """Sends a special birthday email with personalized offers."""
    with app.app_context():
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            email_logger.error(f"User not found for sending birthday email: {user_id}")
            return
            
        if not birthday_data or not birthday_data.get('offers'):
            email_logger.info(f"No birthday data or offers for user {user_id}. Skipping email.")
            return

        html_content = get_birthday_email_template(user['name'], birthday_data)
        send_email(user['email'], f"🎂 Happy Birthday, {user['name']}! Special Offers Inside!", html_content)

def schedule_email_trigger(user_id, email_type, delay_seconds):
    """Schedules an email to be sent after a delay using threading."""
    if email_type == 'cart':
        target_func = send_cart_email
    elif email_type == 'events':
        target_func = send_events_email
    else:
        return

    trigger_time = datetime.now() + timedelta(seconds=delay_seconds)
    email_triggers_collection.insert_one({
        'user_id': user_id,
        'email_type': email_type,
        'trigger_time': trigger_time,
        'status': 'scheduled',
        'created_at': datetime.now()
    })
    
    # Use threading.Timer for a simple delayed execution
    threading.Timer(delay_seconds, target_func, args=[user_id]).start()
    email_logger.info(f"Scheduled '{email_type}' email for user {user_id} in {delay_seconds / 3600:.1f} hours.")

# =================== OBJECTID SERIALIZATION HELPER ===================

def serialize_objectid(obj):
    """Convert ObjectId to string recursively in dictionaries and lists"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_objectid(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_objectid(item) for item in obj]
    else:
        return obj

def ensure_objectid(value):
    """Convert string to ObjectId if needed"""
    if isinstance(value, str):
        try:
            return ObjectId(value)
        except:
            return value
    return value

# =================== DIRECT IMAGE URL PROCESSING ===================

def get_direct_image_url(image_url):
    """DIRECT image URL processing without fallbacks"""
    if not image_url:
        return None
    
    # If already a full URL, return as-is
    if image_url.startswith('http'):
        return image_url
    
    # Clean and build Company URL
    clean_url = image_url.strip().lstrip('/')
    if clean_url:
        return f"https://s7d9.scene7.com/is/image/dollargeneral/{clean_url}"
    
    return None

# =================== USER AUTHENTICATION FUNCTIONS ===================

def hash_password(password):
    """Return password as plain text (no hashing)"""
    return password

def check_password(password, stored_password):
    """Check password against stored plain text password"""
    return password == stored_password

def calculate_age_from_birthday(birthday):
    """Calculate age from birthday date"""
    try:
        if isinstance(birthday, str):
            # Parse birthday string in format YYYY-MM-DD
            birth_date = datetime.strptime(birthday, "%Y-%m-%d")
        elif isinstance(birthday, datetime):
            birth_date = birthday
        else:
            return None
            
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    except Exception as e:
        logger.error(f"Error calculating age from birthday: {e}")
        return None

def create_user(name, email, password, location, birthday, gender):
    """Create a new user account with date of birth"""
    try:
        # Check if user already exists
        existing_user = users_collection.find_one({'email': email})
        if existing_user:
            return False, "User with this email already exists"
        
        # Store password as plain text
        plain_password = hash_password(password)
        
        # Parse the birthday string and calculate age
        try:
            birthday_date = datetime.strptime(birthday, "%Y-%m-%d")
            calculated_age = calculate_age_from_birthday(birthday)
            
            if calculated_age is None or calculated_age < 13:
                return False, "You must be at least 13 years old to register"
                
        except ValueError:
            return False, "Invalid date format for birthday"
        
        # Create user document
        user_doc = {
            'name': name,
            'email': email,
            'password': plain_password,
            'location': location,
            'birthday': birthday_date,
            'age': calculated_age,
            'gender': gender,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        # Insert user
        result = users_collection.insert_one(user_doc)
        
        logger.info(f"Created new user: {email}, Age: {calculated_age}")
        return True, str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return False, str(e)

def authenticate_user(email, password):
    """Authenticate user with email and password"""
    try:
        # Find user by email
        user = users_collection.find_one({'email': email})
        if not user:
            return False, None
        
        # Check password
        if check_password(password, user['password']):
            # Remove password from user data before returning
            user.pop('password', None)
            user['_id'] = str(user['_id'])
            
            # Recalculate age from birthday if needed
            if 'birthday' in user and user['birthday']:
                current_age = calculate_age_from_birthday(user['birthday'])
                if current_age:
                    user['age'] = current_age
                    # Update age in database
                    users_collection.update_one(
                        {'_id': ObjectId(user['_id'])},
                        {'$set': {'age': current_age, 'updated_at': datetime.now()}}
                    )
            
            return True, user
        else:
            return False, None
            
    except Exception as e:
        logger.error(f"Error authenticating user: {e}")
        return False, None

# =================== NEW: BIRTHDAY CELEBRATION FUNCTIONS ===================

def is_birthday_week(birthday_date):
    """Check if current date is within user's birthday week"""
    try:
        if not birthday_date:
            return False
        
        # Handle both datetime and string formats
        if isinstance(birthday_date, str):
            birthday_date = datetime.strptime(birthday_date, "%Y-%m-%d")
        
        today = datetime.now()
        
        # Create this year's birthday
        current_year_birthday = birthday_date.replace(year=today.year)
        
        # Check if today is within 3 days before to 3 days after birthday
        days_diff = (today - current_year_birthday).days
        
        # Birthday week: 3 days before to 3 days after
        return -3 <= days_diff <= 3
        
    except Exception as e:
        birthday_logger.error(f"Error checking birthday week: {e}")
        return False

def is_exact_birthday(birthday_date):
    """Check if today is the exact birthday"""
    try:
        if isinstance(birthday_date, str):
            birthday_date = datetime.strptime(birthday_date, "%Y-%m-%d")
        
        today = datetime.now()
        current_year_birthday = birthday_date.replace(year=today.year)
        
        return today.date() == current_year_birthday.date()
        
    except Exception as e:
        return False

def get_days_until_birthday(birthday_date):
    """Get days until birthday (for messaging)"""
    try:
        if isinstance(birthday_date, str):
            birthday_date = datetime.strptime(birthday_date, "%Y-%m-%d")
        
        today = datetime.now()
        current_year_birthday = birthday_date.replace(year=today.year)
        
        days_diff = (current_year_birthday - today).days
        
        if days_diff < 0:  # Birthday passed this year
            next_year_birthday = birthday_date.replace(year=today.year + 1)
            days_diff = (next_year_birthday - today).days
        
        return days_diff
        
    except Exception as e:
        return None

def get_birthday_celebration(user_id):
    """Get birthday celebration data if user is in birthday week"""
    try:
        birthday_logger.info(f"Checking birthday celebration for user {user_id}")
        
        # Get user data
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user or not user.get('birthday'):
            birthday_logger.info(f"No birthday data found for user {user_id}")
            return None
        
        birthday_date = user['birthday']
        user_name = user['name']

        first_name = user_name.split()[0] if user_name else "User"
        user_age = calculate_age_from_birthday(birthday_date)
        
        # Check if it's birthday week
        if not is_birthday_week(birthday_date):
            birthday_logger.info(f"Not birthday week for {first_name} - birthday is {birthday_date}")
            return None
        
        birthday_logger.info(f"🎂 Birthday week detected for {user_name}!")
        
        # Get user preferences for personalized offers
        user_prefs = get_user_preferences(user_id)
        
        # Get birthday offers based on user's preferences
        birthday_offers = get_birthday_offers(user_id, user_prefs)
        
        # Determine birthday status
        is_today = is_exact_birthday(birthday_date)
        days_until = get_days_until_birthday(birthday_date)
        
        # Create birthday messages
        if is_today:
            celebration_message = f"🎉 Happy Birthday {first_name}! 🎂"
            celebration_subtitle = f"🎈 It's your special day! Celebrating your {user_age}th birthday!"
        else:
            days_diff = (datetime.now() - birthday_date.replace(year=datetime.now().year)).days
            if days_diff > 0:
                celebration_message = f"🎉 Happy Birthday Week {first_name}! 🎂"
                celebration_subtitle = f"🎈 Still celebrating your {user_age}th birthday week!"
            else:
                celebration_message = f"🎉 Birthday Week Special {first_name}! 🎂"
                celebration_subtitle = f"🎈 Your {user_age}th birthday is coming up!"
        
        # Create birthday celebration data
        celebration_data = {
            'user_name': user_name,
            'user_age': user_age,
            'birthday_date': birthday_date.strftime('%B %d') if isinstance(birthday_date, datetime) else birthday_date,
            'celebration_message': celebration_message,
            'celebration_subtitle': celebration_subtitle,
            'offers': birthday_offers,
            'is_birthday_today': is_today,
            'days_until_birthday': days_until,
            'birthday_week_end': (birthday_date.replace(year=datetime.now().year) + timedelta(days=3)).strftime('%B %d')
        }
        
        birthday_logger.info(f"✅ Birthday celebration created for {user_name} with {len(birthday_offers)} offers")
        return celebration_data
        
    except Exception as e:
        birthday_logger.error(f"Error getting birthday celebration: {e}")
        return None

def get_birthday_offers(user_id, user_prefs, limit=6):
    """Get personalized birthday offers based on user preferences"""
    try:
        birthday_logger.info(f"Creating birthday offers for user {user_id}")
        
        # Get user's favorite categories/subcategories
        selected_subcategories = user_prefs.get('selected_subcategories', [])
        liked_products = user_prefs.get('liked_products', [])
        
        birthday_offers = []
        used_product_ids = set()
        
        # Strategy 1: Get products from user's favorite subcategories
        if selected_subcategories and len(birthday_offers) < limit:
            birthday_logger.info(f"Getting offers from user's favorite subcategories: {selected_subcategories}")
            query = {'sub_category': {'$in': selected_subcategories}}
            preferred_products = list(products_collection.find(query).limit(limit * 2))
            
            for product in preferred_products:
                if len(birthday_offers) >= limit:
                    break
                if str(product['_id']) not in used_product_ids:
                    offer = create_birthday_offer(product, discount_percentage=random.randint(20, 35))
                    if offer:
                        birthday_offers.append(offer)
                        used_product_ids.add(str(product['_id']))
        
        # Strategy 2: If not enough offers, get products similar to liked ones
        if len(birthday_offers) < limit and liked_products:
            birthday_logger.info(f"Getting offers similar to liked products")
            for product_id in liked_products[:3]:
                if len(birthday_offers) >= limit:
                    break
                try:
                    liked_product = products_collection.find_one({'_id': ensure_objectid(product_id)})
                    if liked_product:
                        # Find similar products in same category
                        similar_query = {
                            'category': liked_product.get('category'),
                            '_id': {'$ne': liked_product['_id']}
                        }
                        similar_products = list(products_collection.find(similar_query).limit(3))
                        
                        for similar_product in similar_products:
                            if len(birthday_offers) >= limit:
                                break
                            if str(similar_product['_id']) not in used_product_ids:
                                offer = create_birthday_offer(similar_product, discount_percentage=random.randint(25, 40))
                                if offer:
                                    birthday_offers.append(offer)
                                    used_product_ids.add(str(similar_product['_id']))
                except Exception as e:
                    birthday_logger.error(f"Error finding similar products: {e}")
                    continue
        
        # Strategy 3: Fill remaining with popular products
        if len(birthday_offers) < limit:
            birthday_logger.info(f"Filling remaining slots with popular products")
            popular_products = list(products_collection.find({'rating': {'$gte': 4.0}}).limit(limit * 2))
            
            for product in popular_products:
                if len(birthday_offers) >= limit:
                    break
                if str(product['_id']) not in used_product_ids:
                    offer = create_birthday_offer(product, discount_percentage=random.randint(15, 30))
                    if offer:
                        birthday_offers.append(offer)
                        used_product_ids.add(str(product['_id']))
        
        birthday_logger.info(f"Created {len(birthday_offers)} birthday offers")
        return birthday_offers
        
    except Exception as e:
        birthday_logger.error(f"Error getting birthday offers: {e}")
        return []

def create_birthday_offer(product, discount_percentage=20):
    """Create a birthday offer from a product"""
    try:
        # Extract price value
        price_str = product.get('price', '$0.00')
        price_match = re.search(r'[\d.]+', price_str)
        original_price = float(price_match.group()) if price_match else 0.0
        
        if original_price <= 0:
            return None
        
        # Calculate discounted price
        discount_amount = original_price * (discount_percentage / 300)
        discounted_price = original_price - discount_amount
        
        offer = {
            'product_id': str(product['_id']),
            'original_price': f"${original_price:.2f}",
            'discount_percentage': discount_percentage,
            'discounted_price': f"${discounted_price:.2f}",
            'offer_description': f"🎂 Birthday Special: {discount_percentage}% off just for you!",
            'product': format_product(product),
            'matches_preferences': True,  # Birthday offers are always personalized
            'is_birthday_offer': True
        }
        
        return offer
        
    except Exception as e:
        birthday_logger.error(f"Error creating birthday offer: {e}")
        return None

# =================== HELPER FUNCTIONS ===================

def get_user_location_from_session_or_db(user_id):
    """Helper function to get user location from session or database"""
    try:
        # Try to get from session first
        if 'location' in session and session['location']:
            return session['location']
        
        # Fallback to database
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if user and user.get('location'):
            return user['location']
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting user location: {e}")
        return None

def debug_demographics_collection(location):
    """Debug function to inspect demographics collection structure"""
    try:
        demographic_logger.info(f"DEBUGGING DEMOGRAPHICS COLLECTION for location: {location}")
        
        # Get sample records
        sample_records = list(demographics_collection.find().limit(30))
        demographic_logger.info(f"Sample records count: {len(sample_records)}")
        
        if sample_records:
            demographic_logger.info("Sample record structure:")
            for i, record in enumerate(sample_records[:3]):
                demographic_logger.info(f"  Record {i+1}: {record}")
        
        # Count records by location
        location_counts = list(demographics_collection.aggregate([
            {'$group': {'_id': '$location', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}},
            {'$limit': 30}
        ]))
        
        demographic_logger.info(f"Top locations in demographics: {location_counts}")
        
        # Check if the specific location exists
        location_records = list(demographics_collection.find({
            'location': {'$regex': location, '$options': 'i'}
        }).limit(5))
        
        demographic_logger.info(f"Records for '{location}': {len(location_records)}")
        
        return len(location_records) > 0
        
    except Exception as e:
        demographic_logger.error(f"Error debugging demographics collection: {e}")
        return False

def enhance_query_analysis_for_location(user_query, user_id):
    """Enhanced query analysis specifically for location-based queries"""
    try:
        query_lower = user_query.lower()
        
        # Detect location-based queries
        location_keywords = [
            'around me', 'near me', 'in my area', 'people around me',
            'people buying around', 'what are people buying around me',
            'popular in my location', 'trending around me'
        ]
        
        is_location_query = any(keyword in query_lower for keyword in location_keywords)
        
        if is_location_query:
            # Get user's location
            user_location = get_user_location_from_session_or_db(user_id)
            
            if user_location:
                logger.info(f"Detected location query for user location: {user_location}")
                
                # Debug demographics collection
                debug_demographics_collection(user_location)
                
                return {
                    'intent': 'demographic_recommendation',
                    'demographics_filter': True,
                    'query_type': 'location_based',
                    'extracted_info': {
                        'demographic_filters': {
                            'location': user_location,
                            'age_range': None,
                            'gender': None
                        }
                    },
                    'action': None,
                    'cart_action': False
                }
            else:
                logger.warning(f"Location query detected but no user location found")
        
        return None  # Return None if not a location query
        
    except Exception as e:
        logger.error(f"Error in enhanced query analysis: {e}")
        return None

# =================== FIXED EVENTS COLLECTION FUNCTIONS ===================

def get_current_events(user_id=None):
    """FIXED: Fetch active events and enrich with product details."""
    try:
        # Use timezone-aware current date
        current_date = datetime.now(timezone.utc)
        logger.info(f"DEBUG: Checking for active events at UTC time: {current_date}")
        
        # Debug: Check what's in the events collection
        total_events = events_collection.count_documents({})
        logger.info(f"DEBUG: Total events in collection: {total_events}")
        
        if total_events == 0:
            logger.warning("No events found in events collection!")
            return []
        
        # Log sample events for debugging
        sample_events = list(events_collection.find().limit(3))
        for i, event in enumerate(sample_events):
            start_date = event.get('start_date')
            end_date = event.get('end_date')
            logger.info(f"Sample event {i+1}: {event.get('event_name')}")
            logger.info(f"  Start: {start_date} (type: {type(start_date)})")
            logger.info(f"  End: {end_date} (type: {type(end_date)})")
        
        # Query for active events with better date handling
        active_events_query = {
            "$and": [
                {"start_date": {"$lte": current_date}},
                {"end_date": {"$gte": current_date}}
            ]
        }
        
        logger.info(f"Querying events with: {active_events_query}")
        active_events = list(events_collection.find(active_events_query))
        
        logger.info(f"Found {len(active_events)} active events")
        
        if len(active_events) == 0:
            # Additional debugging - check date formats manually
            all_events = list(events_collection.find())
            logger.info("DEBUG: Manual date checking for troubleshooting:")
            for event in all_events:
                start_date = event.get('start_date')
                end_date = event.get('end_date')
                logger.info(f"Event: {event.get('event_name')}")
                
                # Check if current date falls within range manually
                if isinstance(start_date, str):
                    try:
                        start_dt = dateutil.parser.parse(start_date)
                        end_dt = dateutil.parser.parse(end_date)
                        is_active = start_dt <= current_date <= end_dt
                        logger.info(f"  String dates - Start: {start_dt}, End: {end_dt}, Active: {is_active}")
                    except Exception as e:
                        logger.error(f"  Date parsing error: {e}")
                elif isinstance(start_date, datetime):
                    is_active = start_date <= current_date <= end_date
                    logger.info(f"  Datetime objects - Active: {is_active}")
                else:
                    logger.error(f"  Unknown date type: {type(start_date)}")
            
            return []
        
        enriched_events = []
        user_prefs = get_user_preferences(user_id) if user_id else None
        
        for event in active_events:
            enriched_offers = []
            for offer in event.get('offers', []):
                product_id = ensure_objectid(offer.get('product_id'))
                product = products_collection.find_one({'_id': product_id})
                
                if product:
                    enriched_offer = {
                        **offer,
                        'product': format_product(product),
                        'matches_preferences': False
                    }
                    if user_prefs:
                        category = product.get('category')
                        sub_category = product.get('sub_category')
                        if (category in user_prefs.get('selected_categories', []) or
                            sub_category in user_prefs.get('selected_subcategories', [])):
                            enriched_offer['matches_preferences'] = True
                    enriched_offers.append(enriched_offer)
            
            if enriched_offers:
                # Ensure dates are properly formatted for frontend
                start_date = event.get('start_date')
                end_date = event.get('end_date')
                
                # Convert to ISO format if they're datetime objects
                if isinstance(start_date, datetime):
                    start_date_iso = start_date.isoformat()
                else:
                    start_date_iso = start_date
                    
                if isinstance(end_date, datetime):
                    end_date_iso = end_date.isoformat()
                else:
                    end_date_iso = end_date
                
                enriched_event = {
                    'event_name': event['event_name'],
                    'event_description': event['event_description'],
                    'banner_image_url': event.get('banner_image_url'),
                    'offers': enriched_offers,
                    'start_date': start_date_iso,
                    'end_date': end_date_iso
                }
                enriched_events.append(enriched_event)
        
        if user_prefs:
            for event in enriched_events:
                event['offers'].sort(key=lambda o: o['matches_preferences'], reverse=True)
        
        logger.info(f"Returning {len(enriched_events)} enriched events")
        return enriched_events
    
    except Exception as e:
        logger.error(f"Error fetching current events: {e}")
        return []

def fix_events_dates():
    """ADDED: Fix existing events to have proper datetime objects"""
    try:
        logger.info("Checking and fixing events date formats...")
        events_fixed = 0
        
        for event in events_collection.find():
            update_doc = {}
            
            # Convert string dates to datetime objects
            if isinstance(event.get('start_date'), str):
                try:
                    parsed_start = dateutil.parser.parse(event['start_date'])
                    update_doc['start_date'] = parsed_start
                    logger.info(f"Converting start_date from string: {event['start_date']} -> {parsed_start}")
                except Exception as e:
                    logger.error(f"Error parsing start_date for event {event.get('event_name')}: {e}")
                
            if isinstance(event.get('end_date'), str):
                try:
                    parsed_end = dateutil.parser.parse(event['end_date'])
                    update_doc['end_date'] = parsed_end
                    logger.info(f"Converting end_date from string: {event['end_date']} -> {parsed_end}")
                except Exception as e:
                    logger.error(f"Error parsing end_date for event {event.get('event_name')}: {e}")
            
            # Update if needed
            if update_doc:
                events_collection.update_one(
                    {'_id': event['_id']}, 
                    {'$set': update_doc}
                )
                events_fixed += 1
                logger.info(f"Updated date formats for event: {event.get('event_name')}")
                
        logger.info(f"Fixed {events_fixed} events with proper datetime objects")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing events dates: {e}")
        return False

def ensure_events_data():
    """ADDED: Ensure events data is properly inserted with correct date formats"""
    try:
        # Check if events exist
        existing_events = events_collection.count_documents({})
        logger.info(f"Existing events in collection: {existing_events}")
        
        if existing_events == 0:
            logger.info("No events found, inserting sample events...")
            
            # # Sample event data with proper date parsing
            # sample_events = [
            #     {
            #         "event_name": "Ganesh Chaturthi Offers",
            #         "event_description": "Celebrate Ganesh Chaturthi with exclusive discounts on smartphones, earbuds, and speakers.",
            #         "start_date": dateutil.parser.parse("2025-08-17T00:00:00Z"),
            #         "end_date": dateutil.parser.parse("2025-09-01T23:59:59Z"),
            #         "banner_image_url": "https://img.freepik.com/free-vector/flat-ganesh-chaturthi-horizontal-banner-template_23-2149505432.jpg",
            #         "offers": [
            #             {
            #                 "product_id": "688daf7932e00504209c01ae",
            #                 "original_price": "$69.00",
            #                 "discount_percentage": 15,
            #                 "discounted_price": "$58.65",
            #                 "offer_description": "15% off on Tracfone Motorola moto g Play Prepaid Smartphone"
            #             },
            #             {
            #                 "product_id": "688daf7932e00504209c01c4",
            #                 "original_price": "$12.75",
            #                 "discount_percentage": 20,
            #                 "discounted_price": "$30.20",
            #                 "offer_description": "Ganesh Chaturthi Special: 20% off on Sentry True Wireless Bluetooth Buds"
            #             },
            #             {
            #                 "product_id": "688daf7932e00504209c01d0",
            #                 "original_price": "$40.00",
            #                 "discount_percentage": 25,
            #                 "discounted_price": "$30.00",
            #                 "offer_description": "Flat 25% off on Craig Tower Speaker System"
            #             }
            #         ]
            #     }
            # ]
            
            # # Insert events
            # result = events_collection.insert_many(sample_events)
            # logger.info(f"Inserted {len(result.inserted_ids)} events")
            
        else:
            # Fix existing events date formats
            logger.info("Events exist, checking date formats...")
            fix_events_dates()
        
        # Verify events after processing
        current_events = get_current_events()
        logger.info(f"After processing: Found {len(current_events)} active events")
        
        return True
        
    except Exception as e:
        logger.error(f"Error ensuring events data: {e}")
        return False

def add_offer_to_product(product, events):
    """Add offer information to product if it's part of an active event"""
    for event in events:
        for offer in event['offers']:
            if str(offer['product']['_id']) == str(product['_id']):
                product['offer'] = {
                    'discounted_price': offer['discounted_price'],
                    'discount_percentage': offer['discount_percentage'],
                    'offer_description': offer['offer_description'],
                    'event_name': event['event_name'],
                    'original_price': offer['original_price']
                }
                return product
    return product

# =================== INITIALIZATION ===================

def initialize_vertex_ai():
    global embeddings_model, llm_model
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    vertex_project = os.getenv("VERTEX_PROJECT_ID", "")
    vertex_location = os.getenv("VERTEX_LOCATION", "us-central1")

    if not credentials_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set")
    if not vertex_project:
        raise ValueError("VERTEX_PROJECT_ID is not set")

    try:
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        vertexai.init(project=vertex_project, location=vertex_location, credentials=credentials)
        
        embeddings_model = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=vertex_project,
            credentials=credentials
        )
        
        llm_model = ChatVertexAI(
            model_name="gemini-2.0-flash",
            project=vertex_project,
            credentials=credentials,
            temperature=0.2,
            max_output_tokens=2048,
            callbacks = [opik_tracer]
        )
        
        logger.info("Vertex AI initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI: {e}")
        raise

def setup_qdrant_collections():
    """Setup Qdrant collections - ONLY for product embeddings"""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        # ONLY create product_embeddings collection
        collections_config = {
            "product_embeddings": {
                "size": 768,
                "description": "Product embeddings for similarity search"
            }
        }
        
        for collection_name, config in collections_config.items():
            if collection_name not in collection_names:
                logger.info(f"Creating collection: {collection_name}")
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=config["size"],
                        distance=models.Distance.COSINE
                    )
                )
                
                # Create indexes
                if collection_name == "product_embeddings":
                    qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="category",
                        field_schema="keyword"
                    )
                    qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name="sub_category",
                        field_schema="keyword"
                    )
                
                logger.info(f"Created collection: {collection_name}")
        
        logger.info("Qdrant collections setup completed")
        
    except Exception as e:
        logger.error(f"Error setting up Qdrant collections: {e}")
        raise

def get_categories_from_db():
    """Get categories from MongoDB products collection"""
    try:
        # Aggregate categories and subcategories with product counts
        pipeline = [
            {
                '$match': {
                    'category': {'$exists': True, '$ne': None, '$ne': ''}
                }
            },
            {
                '$group': {
                    '_id': {
                        'category': '$category',
                        'sub_category': '$sub_category'
                    },
                    'count': {'$sum': 1}
                }
            },
            {
                '$group': {
                    '_id': '$_id.category',
                    'subcategories': {
                        '$push': '$_id.sub_category'
                    },
                    'total_count': {'$sum': '$count'}
                }
            },
            {
                '$sort': {'total_count': -1}
            }
        ]
        
        result = list(products_collection.aggregate(pipeline))
        
        if not result:
            return get_fallback_categories()
        
        categories = {}
        for item in result:
            category_name = item['_id']
            if category_name and category_name.strip():
                # Clean up subcategories
                subcategories = []
                for sub in item['subcategories']:
                    if sub and sub.strip() and sub not in subcategories:
                        subcategories.append(sub.strip())
                
                categories[category_name] = {
                    'name': category_name,
                    'subcategories': sorted(subcategories),
                    'count': item['total_count']
                }
        
        return categories if categories else get_fallback_categories()
        
    except Exception as e:
        logger.error(f"Error getting categories from database: {e}")
        return get_fallback_categories()

def get_fallback_categories():
    """Return basic fallback categories"""
    return {
        'Electronics': {
            'name': 'Electronics',
            'subcategories': ['Gaming', 'Batteries', 'Phones', 'Portable Audio', 'Computers', 'Accessories'],
            'count': 75
        },
        'Beauty': {
            'name': 'Beauty',
            'subcategories': ['Cosmetics', 'Hair Care', 'Skin Care', 'Soap & Body Wash', 'Nail Care', 'Fragrance'],
            'count': 79
        },
        'Food & Beverages': {
            'name': 'Food & Beverages',
            'subcategories': ['Baby Food', 'Cereals And Breakfast', 'Coffee And Tea', 'Frozen Foods', 'Snacks', 'Beverages'],
            'count': 74
        }
    }

# =================== ENHANCED USER INTERACTIONS SYSTEM ===================

def store_user_interaction(user_id, product_id, interaction_type, context=""):
    """Store interaction in BOTH user_interactions collection AND user_preferences"""
    try:
        interactions_logger.info(f"STORING INTERACTION: User {user_id}, Product {product_id}, Type {interaction_type}")
        
        # Get product details
        try:
            product_oid = ensure_objectid(product_id)
            product = products_collection.find_one({'_id': product_oid})
            if not product:
                interactions_logger.error(f"Product not found: {product_id}")
                return False, None
        except Exception as e:
            interactions_logger.error(f"Error finding product {product_id}: {e}")
            return False, None
        
        current_time = datetime.now()
        
        # 1. STORE IN user_interactions collection (PERSISTENT)
        interaction_doc = {
            'user_id': user_id,
            'product_id': str(product['_id']),
            'interaction_type': interaction_type,
            'product_snapshot': {
                'name': product.get('name', ''),
                'category': product.get('category', ''),
                'sub_category': product.get('sub_category', ''),
                'price': product.get('price', ''),
                'brand': product.get('brand', ''),
                'image_url': product.get('image_url', '')
            },
            'timestamp': current_time,
            'context': context,
            'session_date': current_time.strftime('%Y-%m-%d')
        }
        
        # Insert into user_interactions collection
        user_interactions_collection.insert_one(interaction_doc)
        interactions_logger.info(f"Stored interaction in user_interactions collection")
        
        # 2. UPDATE user_preferences collection with scoring
        prefs = get_user_preferences(user_id)
        
        # Initialize lists if they don't exist
        for key in ['liked_products', 'cart_products', 'disliked_products', 'recent_interactions']:
            if key not in prefs:
                prefs[key] = []
        
        product_id_str = str(product['_id'])
        
        # Update preference lists based on interaction type
        if interaction_type == 'like':
            if product_id_str not in prefs['liked_products']:
                prefs['liked_products'].append(product_id_str)
            # Remove from disliked if present
            if product_id_str in prefs['disliked_products']:
                prefs['disliked_products'].remove(product_id_str)
            interactions_logger.info(f"Added to liked_products: {product.get('name', '')}")
                
        elif interaction_type == 'dislike':
            if product_id_str not in prefs['disliked_products']:
                prefs['disliked_products'].append(product_id_str)
            # Remove from liked and cart if present
            if product_id_str in prefs['liked_products']:
                prefs['liked_products'].remove(product_id_str)
            if product_id_str in prefs['cart_products']:
                prefs['cart_products'].remove(product_id_str)
            interactions_logger.info(f"Added to disliked_products: {product.get('name', '')}")
                
        elif interaction_type == 'cart':
            if product_id_str not in prefs['cart_products']:
                prefs['cart_products'].append(product_id_str)
            # Remove from disliked if present
            if product_id_str in prefs['disliked_products']:
                prefs['disliked_products'].remove(product_id_str)
            interactions_logger.info(f"Added to cart_products: {product.get('name', '')}")
        
        # Add to recent interactions (keep last 50)
        recent_interaction = {
            'product_id': product_id_str,
            'type': interaction_type,
            'timestamp': current_time,
            'product_name': product.get('name', ''),
            'category': product.get('category', ''),
            'sub_category': product.get('sub_category', '')
        }
        
        prefs['recent_interactions'].insert(0, recent_interaction)
        prefs['recent_interactions'] = prefs['recent_interactions'][:50]  # Keep only last 50
        
        # Update metadata
        prefs['updated_at'] = current_time
        prefs['last_interaction'] = current_time
        
        # Calculate preference scores
        prefs['preference_scores'] = calculate_preference_scores(prefs)
        
        # Update user_preferences collection
        user_preferences_collection.update_one(
            {'user_id': user_id},
            {'$set': prefs},
            upsert=True
        )
        
        preferences_logger.info(f"Updated user_preferences with {interaction_type} interaction")
        
        # Return product info for UI updates
        product_info = {
            '_id': str(product['_id']),
            'name': product.get('name', ''),
            'price': product.get('price', '$0.00'),
            'image_url': product.get('image_url', ''),
            'category': product.get('category', ''),
            'sub_category': product.get('sub_category', ''),
            'brand': product.get('brand', ''),
            'rating': product.get('rating', 4.0),
            'review_count': product.get('review_count', 0)
        }
        
        interactions_logger.info(f"INTERACTION STORED SUCCESSFULLY: {interaction_type} for {product.get('name', '')}")
        
        return True, product_info
        
    except Exception as e:
        interactions_logger.error(f"Error storing user interaction: {e}")
        return False, None

def calculate_preference_scores(prefs):
    """Calculate preference scores based on interactions"""
    try:
        # Define scoring weights
        weights = {
            'like': 3.0,
            'cart': 2.5,
            'view': 1.0,
            'dislike': -2.0
        }
        
        category_scores = defaultdict(float)
        subcategory_scores = defaultdict(float)
        brand_scores = defaultdict(float)
        
        # Score based on recent interactions
        for interaction in prefs.get('recent_interactions', []):
            interaction_type = interaction.get('type', 'view')
            weight = weights.get(interaction_type, 1.0)
            category = interaction.get('category', '')
            subcategory = interaction.get('sub_category', '')
            
            if category:
                category_scores[category] += weight
            if subcategory:
                subcategory_scores[subcategory] += weight
        
        return {
            'categories': dict(category_scores),
            'subcategories': dict(subcategory_scores),
            'brands': dict(brand_scores),
            'total_interactions': len(prefs.get('recent_interactions', [])),
            'liked_count': len(prefs.get('liked_products', [])),
            'cart_count': len(prefs.get('cart_products', [])),
            'disliked_count': len(prefs.get('disliked_products', [])),
            'last_calculated': datetime.now()
        }
        
    except Exception as e:
        preferences_logger.error(f"Error calculating preference scores: {e}")
        return {}

def get_user_preferences(user_id):
    """Get user preferences from database"""
    try:
        # Get preferences from database
        prefs = user_preferences_collection.find_one({'user_id': user_id})
        
        current_time = datetime.now()
        
        if not prefs:
            # Create new preferences
            prefs = create_default_preferences(user_id)
            user_preferences_collection.insert_one(prefs)
            preferences_logger.info(f"Created new preferences for user {user_id}")
        else:
            # Update last activity
            user_preferences_collection.update_one(
                {'user_id': user_id},
                {'$set': {'last_activity': current_time}}
            )
        
        
        
        return prefs
        
    except Exception as e:
        preferences_logger.error(f"Error getting user preferences: {e}")
        return create_default_preferences(user_id)

def create_default_preferences(user_id):
    """Create default preferences structure"""
    current_time = datetime.now()
    return {
        'user_id': user_id,
        'liked_products': [],
        'cart_products': [],
        'disliked_products': [],
        'recent_interactions': [],
        'selected_categories': [],
        'selected_subcategories': [],
        'preference_scores': {},
        'created_at': current_time,
        'updated_at': current_time,
        'last_activity': current_time,
        'last_interaction': None
    }

def get_user_interactions_history(user_id, days=30):
    """Get user's interaction history from user_interactions collection"""
    try:
        interactions_logger.info(f"Fetching interaction history for user {user_id} (last {days} days)")
        
        # Calculate date threshold
        date_threshold = datetime.now() - timedelta(days=days)
        
        # Query user_interactions collection
        pipeline = [
            {'$match': {
                'user_id': user_id,
                'timestamp': {'$gte': date_threshold}
            }},
            {'$sort': {'timestamp': -1}},
            {'$limit': 3000}  # Limit to prevent memory issues
        ]
        
        interactions = list(user_interactions_collection.aggregate(pipeline))
        
        interactions_logger.info(f"Found {len(interactions)} interactions for user {user_id}")
        
        return interactions
        
    except Exception as e:
        interactions_logger.error(f"Error getting user interaction history: {e}")
        return []

# =================== FIXED OTHER USERS INTERACTIONS ===================

def debug_user_interactions_collection():
    """Debug function to inspect user_interactions collection"""
    try:
        other_users_logger.info("DEBUG: Inspecting user_interactions collection")
        
        # Check if collection exists and has data
        total_count = user_interactions_collection.count_documents({})
        other_users_logger.info(f"DEBUG: Total interactions in collection: {total_count}")
        
        if total_count == 0:
            other_users_logger.warning("DEBUG: user_interactions collection is EMPTY!")
            return False
        
        # Get sample of user_ids
        sample_interactions = list(user_interactions_collection.find().limit(30))
        user_ids = [interaction.get('user_id') for interaction in sample_interactions]
        unique_users = list(set(user_ids))
        
        other_users_logger.info(f"DEBUG: Sample user_ids found: {unique_users}")
        other_users_logger.info(f"DEBUG: Total unique users in sample: {len(unique_users)}")
        
        # Check recent interactions (last 30 days)
        date_threshold = datetime.now() - timedelta(days=30)
        recent_count = user_interactions_collection.count_documents({
            'timestamp': {'$gte': date_threshold}
        })
        other_users_logger.info(f"DEBUG: Recent interactions (30 days): {recent_count}")
        
        return total_count > 0
        
    except Exception as e:
        other_users_logger.error(f"DEBUG: Error inspecting collection: {e}")
        return False

def create_sample_interactions_if_needed():
    """Create sample interactions if the collection is empty (for testing)"""
    try:
        # Check if we need sample data
        total_count = user_interactions_collection.count_documents({})
        if total_count > 0:
            return 0
        
        other_users_logger.info("CREATING sample interactions for testing")
        
        # Get some sample products
        sample_products = list(products_collection.find().limit(30))
        if not sample_products:
            other_users_logger.warning("No products found to create sample interactions")
            return 0
        
        # Create sample users and interactions
        sample_users = ['sample_user_1', 'sample_user_2', 'sample_user_3']
        sample_interactions = []
        
        for user_id in sample_users:
            for product in sample_products[:5]:  # Each user interacts with 5 products
                interaction = {
                    'user_id': user_id,
                    'product_id': str(product['_id']),
                    'interaction_type': random.choice(['like', 'cart', 'view']),
                    'product_snapshot': {
                        'name': product.get('name', ''),
                        'category': product.get('category', ''),
                        'sub_category': product.get('sub_category', ''),
                        'price': product.get('price', ''),
                        'brand': product.get('brand', ''),
                        'image_url': product.get('image_url', '')
                    },
                    'timestamp': datetime.now() - timedelta(days=random.randint(1, 15)),
                    'context': 'sample_data',
                    'session_date': (datetime.now() - timedelta(days=random.randint(1, 15))).strftime('%Y-%m-%d')
                }
                sample_interactions.append(interaction)
        
        # Insert sample interactions
        if sample_interactions:
            user_interactions_collection.insert_many(sample_interactions)
            other_users_logger.info(f"Created {len(sample_interactions)} sample interactions")
            return len(sample_interactions)
        
        return 0
        
    except Exception as e:
        other_users_logger.error(f"Error creating sample interactions: {e}")
        return 0

def get_all_other_users(current_user_id, days=30):
    """FIXED: Get list of all other users who have interactions"""
    try:
        other_users_logger.info(f"Getting all other users excluding {current_user_id}")
        
        # First debug the collection
        if not debug_user_interactions_collection():
            other_users_logger.warning("No data in user_interactions collection")
            # Try to create sample data
            try:
                sample_count = create_sample_interactions_if_needed()
                if sample_count > 0:
                    other_users_logger.info(f"Created {sample_count} sample interactions")
                else:
                    return []
            except Exception as e:
                other_users_logger.error(f"Error creating sample data: {e}")
                return []
        
        # Calculate date threshold
        date_threshold = datetime.now() - timedelta(days=days)
        other_users_logger.info(f"Looking for interactions after {date_threshold}")
        
        # FIXED: More robust aggregation pipeline
        pipeline = [
            {
                '$match': {
                    'user_id': {'$ne': str(current_user_id), '$exists': True, '$ne': None},
                    'timestamp': {'$gte': date_threshold},
                    'interaction_type': {'$in': ['like', 'cart', 'view', 'dislike']}
                }
            },
            {
                '$group': {
                    '_id': '$user_id',
                    'interaction_count': {'$sum': 1},
                    'latest_interaction': {'$max': '$timestamp'},
                    'interaction_types': {'$push': '$interaction_type'},
                    'positive_interactions': {
                        '$sum': {
                            '$cond': [
                                {'$in': ['$interaction_type', ['like', 'cart']]},
                                1,
                                0
                            ]
                        }
                    }
                }
            },
            {
                '$match': {
                    'interaction_count': {'$gte': 1},  # At least 1 interaction
                    'positive_interactions': {'$gte': 1}  # At least 1 positive interaction
                }
            },
            {
                '$sort': {'positive_interactions': -1, 'interaction_count': -1}
            },
            {
                '$limit': 50  # Limit to top 50 other users
            }
        ]
        
        other_users_logger.info("Executing aggregation pipeline")
        other_users_data = list(user_interactions_collection.aggregate(pipeline))
        
        other_users_logger.info(f"Found {len(other_users_data)} other active users")
        
        # Log details about found users
        for i, user_data in enumerate(other_users_data[:5]):
            other_users_logger.info(f"User {i+1}: {user_data['_id']} - {user_data['interaction_count']} interactions, {user_data['positive_interactions']} positive")
        
        return other_users_data
        
    except Exception as e:
        other_users_logger.error(f"Error getting other users: {e}")
        return []

def get_other_users_interactions_by_user(current_user_id, other_users_data, query_category=None, query_subcategory=None, limit=30):
    """FIXED: Enhanced function to get other users' interactions"""
    try:
        other_users_logger.info("ENHANCED OTHER USERS INTERACTIONS")
        other_users_logger.info(f"Current User: {current_user_id}")
        other_users_logger.info(f"Category Filter: {query_category}, Subcategory Filter: {query_subcategory}")
        other_users_logger.info(f"Processing {len(other_users_data)} other users")
        
        if not other_users_data:
            other_users_logger.warning("No other users data provided")
            return []
        
        # Get current user's preferences to avoid showing disliked products
        try:
            current_user_prefs = get_user_preferences(current_user_id)
            current_user_dislikes = set(current_user_prefs.get('disliked_products', []))
            other_users_logger.info(f"Current user has {len(current_user_dislikes)} disliked products")
        except Exception as e:
            other_users_logger.error(f"Error getting current user preferences: {e}")
            current_user_dislikes = set()
        
        # Dictionary to store product interaction data
        product_interaction_data = defaultdict(lambda: {
            'users': set(),
            'positive_interactions': 0,
            'total_interactions': 0,
            'latest_interaction': None,
            'categories': set(),
            'subcategories': set()
        })
        
        # Process each other user's interactions
        for user_data in other_users_data:
            other_user_id = user_data['_id']
            
            other_users_logger.info(f"Processing user {other_user_id} with {user_data['interaction_count']} interactions")
            
            # Build query for this user's interactions
            user_interactions_query = {
                'user_id': other_user_id,
                'timestamp': {'$gte': datetime.now() - timedelta(days=30)},
                'interaction_type': {'$in': ['like', 'cart', 'view']}  # Only positive interactions
            }
            
            # Add category/subcategory filtering if specified
            if query_category or query_subcategory:
                category_filter = {}
                if query_category:
                    category_filter['product_snapshot.category'] = {'$regex': query_category, '$options': 'i'}
                if query_subcategory:
                    category_filter['product_snapshot.sub_category'] = {'$regex': query_subcategory, '$options': 'i'}
                
                if category_filter:
                    user_interactions_query.update(category_filter)
            
            try:
                user_interactions = list(user_interactions_collection.find(user_interactions_query))
                other_users_logger.info(f"Found {len(user_interactions)} filtered interactions for user {other_user_id}")
                
                # Process each interaction
                for interaction in user_interactions:
                    product_id = interaction.get('product_id')
                    interaction_type = interaction.get('interaction_type')
                    product_snapshot = interaction.get('product_snapshot', {})
                    timestamp = interaction.get('timestamp')
                    
                    if not product_id or product_id in current_user_dislikes:
                        continue
                    
                    # Update product interaction data
                    product_data = product_interaction_data[product_id]
                    product_data['users'].add(other_user_id)
                    product_data['total_interactions'] += 1
                    
                    if interaction_type in ['like', 'cart']:
                        product_data['positive_interactions'] += 1
                    
                    if not product_data['latest_interaction'] or timestamp > product_data['latest_interaction']:
                        product_data['latest_interaction'] = timestamp
                    
                    # Store category/subcategory info
                    if product_snapshot.get('category'):
                        product_data['categories'].add(product_snapshot['category'])
                    if product_snapshot.get('sub_category'):
                        product_data['subcategories'].add(product_snapshot['sub_category'])
                        
            except Exception as e:
                other_users_logger.error(f"Error processing interactions for user {other_user_id}: {e}")
                continue
        
        other_users_logger.info(f"Processed interactions for {len(product_interaction_data)} unique products")
        
        if not product_interaction_data:
            other_users_logger.warning("No product interaction data found")
            return []
        
        # Filter and score products - FIXED: More lenient filtering
        scored_products = []
        for product_id, data in product_interaction_data.items():
            # FIXED: More lenient filtering - at least 1 user and 1 positive interaction
            if len(data['users']) >= 1 and data['positive_interactions'] >= 1:
                score = (data['positive_interactions'] * 2) + len(data['users'])
                
                scored_products.append({
                    'product_id': product_id,
                    'score': score,
                    'users_count': len(data['users']),
                    'positive_interactions': data['positive_interactions'],
                    'total_interactions': data['total_interactions'],
                    'latest_interaction': data['latest_interaction']
                })
        
        # Sort by score and limit results
        scored_products.sort(key=lambda x: x['score'], reverse=True)
        top_products = scored_products[:limit]
        
        other_users_logger.info(f"Top {len(top_products)} products by other users")
        
        # Fetch full product details
        other_user_products = []
        for item in top_products:
            try:
                product = products_collection.find_one({'_id': ensure_objectid(item['product_id'])})
                
                if product:
                    product_data = {
                        '_id': str(product['_id']),
                        'name': product.get('name', ''),
                        'price': product.get('price', '$0.00'),
                        'image_url': product.get('image_url', ''),
                        'product_url': product.get('product_url', ''),
                        'category': product.get('category', ''),
                        'sub_category': product.get('sub_category', ''),
                        'description': product.get('description', ''),
                        'brand': product.get('brand', ''),
                        'rating': product.get('rating', 4.0),
                        'review_count': product.get('review_count', 0),
                        'interaction_count': item['total_interactions'],
                        'users_count': item['users_count'],
                        'positive_interactions': item['positive_interactions'],
                        'similarity_score': 0.8,
                        'other_users_score': item['score']
                    }
                    other_user_products.append(product_data)
                    
                    other_users_logger.info(f"Added {product.get('name', 'Unknown')} - {item['positive_interactions']} positive interactions by {item['users_count']} users")
                    
            except Exception as e:
                other_users_logger.error(f"Error fetching product {item['product_id']}: {e}")
                continue
        
        other_users_logger.info(f"FINAL RESULT: {len(other_user_products)} products from other users")
        return other_user_products
        
    except Exception as e:
        other_users_logger.error(f"Error in enhanced other users interactions: {e}")
        return []

def get_other_users_interactions(user_id, query_category=None, query_subcategory=None, limit=30):
    """FIXED: Main function that combines user discovery and interaction processing"""
    try:
        other_users_logger.info("STARTING ENHANCED OTHER USERS INTERACTIONS PROCESS")
        
        # Step 1: Get all other users with FIXED function
        other_users_data = get_all_other_users(user_id, days=30)
        
        if not other_users_data:
            other_users_logger.warning("No other users found after trying all methods")
            return []
        
        # Step 2: Process their interactions with FIXED function
        other_user_products = get_other_users_interactions_by_user(
            user_id, 
            other_users_data, 
            query_category, 
            query_subcategory, 
            limit
        )
        
        other_users_logger.info(f"ENHANCED OTHER USERS PROCESS COMPLETE: {len(other_user_products)} products")
        return other_user_products
        
    except Exception as e:
        other_users_logger.error(f"Error in main other users interactions: {e}")
        return []

# =================== ENHANCED DEMOGRAPHIC RECOMMENDATIONS ===================

def get_products_by_location(user_location, limit=50):
    """Get ALL products that people in the same location are buying (across ALL categories)"""
    try:
        demographic_logger.info(f"Getting products for location: {user_location}")
        
        # Query demographics collection for same location
        location_demographics = list(demographics_collection.find({
            'location': {'$regex': user_location, '$options': 'i'}
        }))
        
        demographic_logger.info(f"Found {len(location_demographics)} demographic records for {user_location}")
        
        if not location_demographics:
            demographic_logger.warning(f"No demographic data found for location: {user_location}")
            return []
        
        # Extract ALL unique product_ids from demographics
        location_product_ids = set()
        category_counts = defaultdict(int)
        
        for demo_record in location_demographics:
            product_id = demo_record.get('product_id')
            if product_id:
                try:
                    # Ensure consistent ObjectId format
                    if isinstance(product_id, str):
                        product_oid = ObjectId(product_id)
                    else:
                        product_oid = product_id
                    
                    location_product_ids.add(product_oid)
                    
                    # Track category distribution
                    if 'category' in demo_record:
                        category_counts[demo_record['category']] += 1
                        
                except Exception as e:
                    demographic_logger.error(f"Error processing product_id {product_id}: {e}")
                    continue
        
        demographic_logger.info(f"Found {len(location_product_ids)} unique products in {user_location}")
        demographic_logger.info(f"Category distribution: {dict(category_counts)}")
        
        # Fetch ALL these products from products collection
        if not location_product_ids:
            return []
        
        # Convert to list for MongoDB query
        product_ids_list = list(location_product_ids)
        
        # Fetch products with proper ObjectId matching
        location_products = list(products_collection.find({
            '_id': {'$in': product_ids_list}
        }).limit(limit))
        
        demographic_logger.info(f"Successfully fetched {len(location_products)} products from database")
        
        # Add demographic popularity data
        enhanced_products = []
        for product in location_products:
            product_oid = product['_id']
            
            # Count how many people in this location bought this product
            popularity_count = len([
                d for d in location_demographics 
                if ensure_objectid(d.get('product_id')) == product_oid
            ])
            
            product_data = {
                '_id': str(product['_id']),
                'name': product.get('name', ''),
                'price': product.get('price', '$0.00'),
                'image_url': product.get('image_url', ''),
                'product_url': product.get('product_url', ''),
                'category': product.get('category', ''),
                'sub_category': product.get('sub_category', ''),
                'description': product.get('description', ''),
                'brand': product.get('brand', ''),
                'rating': product.get('rating', 4.0),
                'review_count': product.get('review_count', 0),
                'demographic_match': True,
                'location_popularity': popularity_count,
                'similarity_score': 0.9  # High score for demographic matches
            }
            enhanced_products.append(product_data)
        
        # Sort by location popularity
        enhanced_products.sort(key=lambda x: x['location_popularity'], reverse=True)
        
        demographic_logger.info(f"TOP PRODUCTS BY LOCATION POPULARITY:")
        for i, product in enumerate(enhanced_products[:30]):
            demographic_logger.info(f"  {i+1}. {product['name']} ({product['category']}) - {product['location_popularity']} people")
        
        return enhanced_products
        
    except Exception as e:
        demographic_logger.error(f"Error getting products by location: {e}")
        return []

def get_enhanced_demographic_products(user_id, demographic_filters):
    """Enhanced demographic products function that handles 'around me' queries properly"""
    try:
        demographic_logger.info(f"ENHANCED DEMOGRAPHIC QUERY - User: {user_id}")
        demographic_logger.info(f"Filters: {demographic_filters}")
        
        location = demographic_filters.get('location')
        age_range = demographic_filters.get('age_range')
        gender = demographic_filters.get('gender')
        
        # Handle location-based queries (e.g., "what are people buying around me?")
        if location:
            demographic_logger.info(f"Processing location-based query for: {location}")
            
            # Get ALL products from this location (across all categories)
            location_products = get_products_by_location(location, limit=300)
            
            if not location_products:
                demographic_logger.warning(f"No products found for location: {location}")
                return [], []
            
            # Apply additional demographic filters if specified
            if age_range or gender:
                demographic_logger.info(f"Applying additional filters - Age: {age_range}, Gender: {gender}")
                
                # Build additional query for demographics collection
                additional_filters = {'location': {'$regex': location, '$options': 'i'}}
                
                if age_range:
                    additional_filters['age'] = {'$gte': age_range[0], '$lte': age_range[1]}
                if gender:
                    additional_filters['gender'] = gender
                
                # Get filtered demographics
                filtered_demographics = list(demographics_collection.find(additional_filters))
                
                if filtered_demographics:
                    # Extract product IDs from filtered demographics
                    filtered_product_ids = set()
                    for demo in filtered_demographics:
                        product_id = demo.get('product_id')
                        if product_id:
                            filtered_product_ids.add(str(ensure_objectid(product_id)))
                    
                    # Filter location_products to only include those in filtered demographics
                    location_products = [
                        p for p in location_products 
                        if p['_id'] in filtered_product_ids
                    ]
                    
                    demographic_logger.info(f"After additional filtering: {len(location_products)} products")
            
            # Split into main recommendations and related products
            main_demographic_products = location_products[:20]  # Top 20 by popularity
            
            # Get related products from same categories
            main_categories = set(p.get('category') for p in main_demographic_products if p.get('category'))
            
            if main_categories:
                # Query for more products in these categories
                related_query = {'category': {'$in': list(main_categories)}}
                related_products_cursor = products_collection.find(related_query).limit(15)
                
                related_products = []
                existing_ids = set(p['_id'] for p in main_demographic_products)
                
                for product in related_products_cursor:
                    if str(product['_id']) not in existing_ids:
                        product_data = {
                            '_id': str(product['_id']),
                            'name': product.get('name', ''),
                            'price': product.get('price', '$0.00'),
                            'image_url': product.get('image_url', ''),
                            'product_url': product.get('product_url', ''),
                            'category': product.get('category', ''),
                            'sub_category': product.get('sub_category', ''),
                            'description': product.get('description', ''),
                            'brand': product.get('brand', ''),
                            'rating': product.get('rating', 4.0),
                            'review_count': product.get('review_count', 0),
                            'similarity_score': 0.7
                        }
                        related_products.append(product_data)
            else:
                related_products = []
            
            demographic_logger.info(f"ENHANCED DEMOGRAPHIC RESULT:")
            demographic_logger.info(f"  Location products: {len(main_demographic_products)}")
            demographic_logger.info(f"  Related products: {len(related_products)}")
            demographic_logger.info(f"  Categories found: {list(main_categories)}")
            
            return main_demographic_products, related_products
        
        else:
            # Fallback to original demographic filtering logic
            demographic_logger.info("Using fallback demographic filtering")
            return [], []
            
    except Exception as e:
        demographic_logger.error(f"Error in enhanced demographic products: {e}")
        return [], []

# =================== CONTINUE SHOPPING WITH HISTORICAL DATA ===================

def get_continue_shopping_products(user_id, limit=6):
    """Get products based on HISTORICAL interactions from user_interactions collection"""
    try:
        preferences_logger.info(f"Getting continue shopping products for user {user_id}")
        
        # Get historical interactions from user_interactions collection
        interactions_history = get_user_interactions_history(user_id, days=30)  # Last 30 days
        
        if not interactions_history:
            preferences_logger.info(f"No historical interactions found for user {user_id}")
            return get_popular_products_fallback(limit)
        
        # Extract product IDs from interactions, prioritizing recent ones
        product_interactions = defaultdict(list)
        for interaction in interactions_history:
            product_id = interaction.get('product_id')
            interaction_type = interaction.get('interaction_type')
            timestamp = interaction.get('timestamp')
            
            if product_id and interaction_type:
                product_interactions[product_id].append({
                    'type': interaction_type,
                    'timestamp': timestamp
                })
        
        # Score products based on interaction types and recency
        scored_products = []
        weights = {'like': 3.0, 'cart': 2.5, 'view': 1.0, 'dislike': -2.0}
        
        for product_id, interactions in product_interactions.items():
            # Calculate score
            total_score = 0
            for interaction in interactions:
                interaction_type = interaction['type']
                weight = weights.get(interaction_type, 1.0)
                
                # Add recency bonus (more recent = higher score)
                days_ago = (datetime.now() - interaction['timestamp']).days
                recency_bonus = max(0, 1 - (days_ago / 30))  # Decay over 30 days
                
                total_score += weight * (1 + recency_bonus)
            
            if total_score > 0:  # Only include positively scored products
                scored_products.append((product_id, total_score))
        
        # Sort by score and get top products
        scored_products.sort(key=lambda x: x[1], reverse=True)
        top_product_ids = [pid for pid, score in scored_products[:limit*2]]
        
        preferences_logger.info(f"Found {len(top_product_ids)} scored products from history")
        
        # Fetch product details
        products = []
        for product_id in top_product_ids:
            try:
                product = products_collection.find_one({'_id': ensure_objectid(product_id)})
                if product:
                    # Get user preferences for reranking
                    user_preferences = get_user_preferences(user_id)
                    product_data = format_product_for_dashboard(product, user_id, user_preferences)
                    products.append(product_data)
            except Exception as e:
                preferences_logger.error(f"Error fetching product {product_id}: {e}")
                continue
        
        # Final reranking with current preferences
        user_preferences = get_user_preferences(user_id)
        reranked_products = enhanced_rerank_with_rating(products, user_preferences)
        
        preferences_logger.info(f"Returning {len(reranked_products[:limit])} continue shopping products")
        return reranked_products[:limit]
        
    except Exception as e:
        preferences_logger.error(f"Error getting continue shopping products: {e}")
        return get_popular_products_fallback(limit)

def get_popular_products_fallback(limit=6):
    """Get popular products as fallback"""
    try:
        pipeline = [
            {'$match': {'rating': {'$gte': 4.0}}},
            {'$sample': {'size': limit}}
        ]
        
        popular_products = list(products_collection.aggregate(pipeline))
        
        products = []
        for product in popular_products:
            product_data = {
                '_id': str(product['_id']),
                'name': product.get('name', ''),
                'price': product.get('price', '$0.00'),
                'image': get_direct_image_url(product.get('image_url', '')),
                'product_url': product.get('product_url', ''),
                'category': product.get('category', ''),
                'sub_category': product.get('sub_category', ''),
                'description': product.get('description', ''),
                'brand': product.get('brand', ''),
                'rating': product.get('rating', 4.0),
                'review_count': product.get('review_count', 0),
                'is_liked': False,
                'is_in_cart': False,
                'matches_preference': False,
                'rerank_score': product.get('rating', 4.0)
            }
            products.append(product_data)
        
        return products
        
    except Exception as e:
        logger.error(f"Error getting popular products fallback: {e}")
        return []

# =================== ENHANCED RERANKING FUNCTIONS ===================

def rerank_products_by_preferences(products, user_preferences):
    """Enhanced function to rerank products based on user preferences"""
    try:
        liked_products = set(user_preferences.get('liked_products', []))
        cart_products = set(user_preferences.get('cart_products', []))
        disliked_products = set(user_preferences.get('disliked_products', []))
        selected_subcategories = set(user_preferences.get('selected_subcategories', []))
        
        # Get preference scores for intelligent reranking
        preference_scores = user_preferences.get('preference_scores', {})
        category_scores = preference_scores.get('categories', {})
        subcategory_scores = preference_scores.get('subcategories', {})
        
        # Define weights
        weights = {
            'like': 3.0,
            'cart': 2.5,
            'subcategory_preference': 2.0,
            'category_preference': 1.5,
            'subcategory_selected': 2.0,
            'dislike': -3.0
        }
        
        scored_products = []
        
        for product in products:
            product_id = str(product.get('_id', ''))
            product_category = product.get('category', '')
            product_subcategory = product.get('sub_category', '')
            
            # Skip disliked products completely
            if product_id in disliked_products:
                continue
            
            # Calculate preference score
            preference_score = 1.0  # Base score
            reasons = []
            
            # User interaction scoring
            if product_id in liked_products:
                preference_score += weights['like']
                reasons.append("You liked this")
                
            if product_id in cart_products:
                preference_score += weights['cart']
                reasons.append("In your cart")
            
            # Selected subcategory scoring
            if product_subcategory in selected_subcategories:
                preference_score += weights['subcategory_selected']
                reasons.append(f"Matches your interest: {product_subcategory}")
            
            # Historical preference scoring
            if product_category in category_scores:
                category_boost = min(category_scores[product_category] / 30, weights['category_preference'])
                preference_score += category_boost
                if category_boost > 0.5:
                    reasons.append(f"You like {product_category}")
            
            if product_subcategory in subcategory_scores:
                subcategory_boost = min(subcategory_scores[product_subcategory] / 5, weights['subcategory_preference'])
                preference_score += subcategory_boost
                if subcategory_boost > 0.5:
                    reasons.append(f"Based on your {product_subcategory} preferences")
            
            # Add metadata
            product['preference_score'] = preference_score
            product['preference_reason'] = '; '.join(reasons) if reasons else ''
            
            scored_products.append(product)
        
        # Sort by preference score, then by similarity score
        scored_products.sort(key=lambda x: (x.get('preference_score', 0), x.get('similarity_score', 0)), reverse=True)
        
        preferences_logger.info(f"Reranked {len(scored_products)} products based on preferences")
        return scored_products
        
    except Exception as e:
        preferences_logger.error(f"Error reranking products: {e}")
        return products

def enhanced_rerank_with_rating(products, user_preferences, boost_subcategories=False):
    """Enhanced reranking combining interaction scores and product ratings"""
    try:
        # First apply preference-based reranking
        reranked_products = rerank_products_by_preferences(products, user_preferences)
        
        # Then apply rating boost
        for product in reranked_products:
            rating = product.get('rating')
            if rating is None or rating == 0:
                rating = 4.0
            
            preference_score = product.get('preference_score', 1.0)
            rating_score = (float(rating) / 5.0) * 1.0  # Rating weight
            
            final_score = preference_score + rating_score
            
            product['rerank_score'] = final_score
            product['rating_score'] = rating_score
        
        # Final sort by combined score
        reranked_products.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        return reranked_products
        
    except Exception as e:
        preferences_logger.error(f"Error in enhanced reranking: {e}")
        return products

# =================== VECTOR SEARCH & EMBEDDING FUNCTIONS ===================

def get_embedding(text):
    """Get embedding for text with error handling"""
    try:
        time.sleep(0.1)
        embedding = embeddings_model.embed_query(text)
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return [0.0] * 768

def vector_similarity_search(query, limit=25):
    """Enhanced vector similarity search on product embeddings"""
    try:
        logger.info(f"Performing vector similarity search for: {query}")
        
        query_embedding = get_embedding(query)
        
        search_results = qdrant_client.query_points(
            collection_name="product_embeddings",
            query=query_embedding,
            limit=limit,
            score_threshold=0.25
        )
        
        similar_products = []
        for result in search_results.points:
            product_id = result.payload.get('product_id')
            if product_id:
                try:
                    product_oid = ensure_objectid(product_id)
                    product = products_collection.find_one({'_id': product_oid})
                    if product:
                        product_data = {
                            '_id': str(product['_id']),
                            'name': product.get('name', ''),
                            'price': product.get('price', '$0.00'),
                            'image_url': product.get('image_url', ''),
                            'product_url': product.get('product_url', ''),
                            'category': product.get('category', ''),
                            'sub_category': product.get('sub_category', ''),
                            'description': product.get('description', ''),
                            'brand': product.get('brand', ''),
                            'rating': product.get('rating'),
                            'review_count': product.get('review_count'),
                            'similarity_score': result.score
                        }
                        similar_products.append(product_data)
                except Exception as e:
                    logger.error(f"Error processing product {product_id}: {e}")
                    continue
        
        logger.info(f"Vector search found {len(similar_products)} products")
        return similar_products
        
    except Exception as e:
        logger.error(f"Error in vector similarity search: {e}")
        return []

# =================== PRODUCT FETCHING FOR RERANKING ===================

def get_all_products_for_reranking(user_preferences, limit=250):
    """Get all products from MongoDB for reranking"""
    try:
        all_products = list(products_collection.find().limit(limit))
        
        formatted_products = []
        for product in all_products:
            product_data = {
                '_id': str(product['_id']),
                'name': product.get('name', ''),
                'price': product.get('price', '$0.00'),
                'image_url': product.get('image_url', ''),
                'product_url': product.get('product_url', ''),
                'category': product.get('category', ''),
                'sub_category': product.get('sub_category', ''),
                'description': product.get('description', ''),
                'brand': product.get('brand', ''),
                'rating': product.get('rating', 4.0),
                'review_count': product.get('review_count', 0),
                'similarity_score': 1.0
            }
            formatted_products.append(product_data)
        
        logger.info(f"Fetched {len(formatted_products)} products for reranking")
        return formatted_products
        
    except Exception as e:
        logger.error(f"Error getting products for reranking: {e}")
        return []

# =================== DASHBOARD PRODUCT FETCHING ===================

def analyze_user_selections(user_preferences):
    """Analyze user selections to determine distribution strategy"""
    selected_categories = user_preferences.get('selected_categories', [])
    selected_subcategories = user_preferences.get('selected_subcategories', [])
    
    if not selected_categories and not selected_subcategories:
        return None
        
    analysis = {
        'has_selections': True,
        'category_selections': selected_categories,
        'subcategory_selections': selected_subcategories,
        'total_selections': len(selected_categories) + len(selected_subcategories),
        'selection_type': 'mixed'
    }
    
    if selected_subcategories and not selected_categories:
        analysis['selection_type'] = 'subcategories_only'
    elif selected_categories and not selected_subcategories:
        analysis['selection_type'] = 'categories_only'
    elif selected_subcategories and selected_categories:
        analysis['selection_type'] = 'mixed'
        
    if len(selected_subcategories) >= 1 and len(set(selected_categories)) <= 2:
        analysis['user_intent'] = 'focused'
    elif len(set(selected_categories)) >= 3 or len(selected_subcategories) >= 5:
        analysis['user_intent'] = 'diverse'
    else:
        analysis['user_intent'] = 'moderate'
        
    return analysis

def calculate_slot_allocation(analysis, max_products=8):
    """Calculate how to distribute product slots based on user selections"""
    if not analysis:
        return []
        
    allocations = []
    
    subcategory_selections = analysis['subcategory_selections']
    category_selections = analysis['category_selections']
    
    if subcategory_selections:
        if len(subcategory_selections) == 1:
            allocations.append({
                'type': 'subcategory',
                'value': subcategory_selections[0],
                'slots': max_products
            })
        elif len(subcategory_selections) <= 4:
            slots_per_sub = max(1, max_products // len(subcategory_selections))
            for subcat in subcategory_selections:
                allocations.append({
                    'type': 'subcategory',
                    'value': subcat,
                    'slots': slots_per_sub
                })
        else:
            for subcat in subcategory_selections[:4]:
                allocations.append({
                    'type': 'subcategory',
                    'value': subcat,
                    'slots': 2
                })
    
    elif category_selections:
        if len(category_selections) == 1:
            allocations.append({
                'type': 'category',
                'value': category_selections[0],
                'slots': max_products
            })
        else:
            slots_per_cat = max(2, max_products // len(category_selections))
            for cat in category_selections:
                allocations.append({
                    'type': 'category',
                    'value': cat,
                    'slots': slots_per_cat
                })
    
    return allocations

def get_products_by_allocation(allocation, limit):
    """Get products for a specific allocation"""
    if allocation['type'] == 'subcategory':
        query = {'sub_category': allocation['value']}
    else:
        query = {'category': allocation['value']}
        
    products_cursor = products_collection.find(query).sort([
        ('rating', -1),
        ('review_count', -1)
    ]).limit(limit * 3)
    
    products = []
    for product in products_cursor:
        rating = product.get('rating')
        if rating is not None and rating >= 3.5:
            products.append(product)
            
    return products[:limit]

def get_enhanced_shopping_trends_products(user_id, max_products=8):
    """Enhanced shopping trends with smart allocation"""
    prefs = get_user_preferences(user_id)
    
    analysis = analyze_user_selections(prefs)
    
    if not analysis:
        return []
        
    allocations = calculate_slot_allocation(analysis, max_products)
    
    if not allocations:
        return []
        
    all_products = []
    for allocation in allocations:
        products = get_products_by_allocation(allocation, allocation['slots'])
        
        for product in products:
            product['allocation_type'] = allocation['type']
            product['allocation_value'] = allocation['value']
            product['selection_reason'] = f"Based on your interest in {allocation['value']}"
            
        all_products.extend(products)
    
    formatted_products = []
    for product in all_products:
        product_data = format_product_for_dashboard(product, user_id, prefs)
        product_data['selection_reason'] = product.get('selection_reason', '')
        formatted_products.append(product_data)
        
    reranked_products = enhanced_rerank_with_rating(formatted_products, prefs, boost_subcategories=True)
    
    return reranked_products[:max_products]


def get_shopping_trends_products(user_id, limit=8):
    """Main function - calls enhanced version"""
    return get_enhanced_shopping_trends_products(user_id, limit)

def format_product_for_dashboard(product, user_id, user_preferences):
    """Format product for dashboard display"""
    try:
        product_data = format_product(product)
        
        # Add interaction indicators
        product_id = str(product['_id'])
        liked_products = user_preferences.get('liked_products', [])
        cart_products = user_preferences.get('cart_products', [])
        selected_subcategories = user_preferences.get('selected_subcategories', [])
        
        # Add interaction states
        product_data['is_liked'] = product_id in liked_products
        product_data['is_in_cart'] = product_id in cart_products
        product_data['matches_preference'] = product.get('sub_category') in selected_subcategories
        
        return product_data
        
    except Exception as e:
        logger.error(f"Error formatting product for dashboard: {e}")
        return format_product(product)

def get_categories_with_subcategories():
    """Get categories with their subcategories AND initial products for display"""
    try:
        # Get top 3 categories by product count with initial products
        pipeline = [
            {'$match': {'category': {'$exists': True, '$ne': None, '$ne': ''}}},
            {'$group': {
                '_id': '$category',
                'subcategories': {'$addToSet': '$sub_category'},
                'count': {'$sum': 1},
                'sample_products': {'$push': {
                    '_id': '$_id',
                    'name': '$name',
                    'price': '$price',
                    'image_url': '$image_url',
                    'category': '$category',
                    'sub_category': '$sub_category',
                    'brand': '$brand',
                    'rating': '$rating',
                    'review_count': '$review_count'
                }}
            }},
            {'$sort': {'count': -1}},
            {'$limit': 3}
        ]
        
        result = list(products_collection.aggregate(pipeline))
        
        categories = {}
        for item in result:
            category_name = item['_id']
            subcategories = [sub for sub in item['subcategories'] if sub and sub.strip()]
            
            # Get 6-8 sample products for initial display
            sample_products = item['sample_products'][:8]
            formatted_products = [format_product(p) for p in sample_products]
            
            categories[category_name] = {
                'name': category_name,
                'subcategories': sorted(subcategories),
                'count': item['count'],
                'icon': get_category_icon(category_name),
                'initial_products': formatted_products
            }
        
        logger.info(f"Retrieved {len(categories)} categories with initial products")
        return categories
        
    except Exception as e:
        logger.error(f"Error getting categories with subcategories: {e}")
        return get_fallback_categories_limited()

def get_category_icon(category_name):
    """Get icon for category"""
    icons = {
        'Electronics': '📱',
        'Beauty': '💄',
        'Food & Beverages': '🍕',
        'Personal Care': '🧴',
        'Home & Garden': '🏠',
        'Clothing': '👕'
    }
    return icons.get(category_name, '📦')


def get_fallback_categories_limited():
    """Get fallback top 3 categories"""
    return {
        'Electronics': {
            'name': 'Electronics',
            'subcategories': ['Gaming', 'Phones', 'Computers', 'Accessories'],
            'count': 75,
            'icon': '📱',
            'initial_products': []
        },
        'Beauty': {
            'name': 'Beauty',
            'subcategories': ['Cosmetics', 'Hair Care', 'Skin Care', 'Fragrance'],
            'count': 79,
            'icon': '💄',
            'initial_products': []
        },
        'Food & Beverages': {
            'name': 'Food & Beverages',
            'subcategories': ['Snacks', 'Beverages', 'Coffee And Tea', 'Frozen Foods'],
            'count': 74,
            'icon': '🍕',
            'initial_products': []
        }
    }


# =================== ENHANCED QUERY ANALYZER ===================
class EnhancedQueryAnalyzer:
    """Enhanced query analyzer with better location detection and FIXED cart parsing"""
    
    def __init__(self):
        self.llm = llm_model
        
    def analyze_query(self, user_query, user_id):
        """Enhanced analyze query with location-specific handling and FIXED cart parsing"""
        try:
            # First check for location-based queries
            location_analysis = enhance_query_analysis_for_location(user_query, user_id)
            if location_analysis:
                return location_analysis
            
            # FIXED: Check for cart actions first with immediate parsing
            if self._is_cart_action(user_query):
                return self._parse_cart_action(user_query, user_id)
            
            # Fall back to original LLM analysis for other queries
            if not self.llm:
                return self._fallback_analysis(user_query, user_id)
                
            analysis_prompt = f"""
You are a query analyzer for a shopping assistant. Analyze the user's query and determine the intent and required processing.

User Query: "{user_query}"

Classify the query into one of these categories:
1. CART_ACTION - queries about cart operations (add to cart, show cart, clear cart, remove from cart)
2. DEMOGRAPHIC_RECOMMENDATION - queries about demographic filtering (people in locations, age groups, gender-based recommendations, "what are people around me buying", "what do people in my age group buy")
3. PRODUCT_RECOMMENDATION - general product search and recommendations

For CART_ACTION, extract:
- action: specific cart action (add_to_cart, show_cart, clear_cart, remove_from_cart)
- product_references: extract any numbers or positions mentioned (e.g., "1,2,3" or "first", "second")
- product_names: extract any specific product names mentioned

For DEMOGRAPHIC_RECOMMENDATION, extract:
- location: if mentioned (or if asking about "around me" use user's location)
- age_range: if mentioned age groups
- gender: if mentioned gender preferences

Respond in this exact JSON format:
{{
    "intent": "CART_ACTION|DEMOGRAPHIC_RECOMMENDATION|PRODUCT_RECOMMENDATION",
    "action": "specific_action_if_applicable",
    "product_references": ["extracted", "numbers", "or", "positions"],
    "product_names": ["extracted", "product", "names"],
    "category": "extracted_category_or_null",
    "demographic_filters": {{
        "location": "extracted_location_or_user_location_if_around_me",
        "age_range": [min_age, max_age] or null,
        "gender": "extracted_gender_or_null"
    }}
}}
"""
            
            response = self.llm.invoke(analysis_prompt)
            try:
                response_text = response.content.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                analysis_result = json.loads(response_text.strip())
                return self._convert_llm_analysis(analysis_result, user_query, user_id)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM analysis response: {e}")
                return self._fallback_analysis(user_query, user_id)
                
        except Exception as e:
            logger.error(f"Error in enhanced query analysis: {e}")
            return self._fallback_analysis(user_query, user_id)
    
    def _is_cart_action(self, user_query):
        """Check if query is a cart action"""
        cart_keywords = ['add to cart', 'cart', 'remove from cart', 'show cart', 'clear cart', 'view cart']
        query_lower = user_query.lower()
        return any(keyword in query_lower for keyword in cart_keywords)
    
    def _parse_cart_action(self, user_query, user_id):
        """FIXED: Parse cart action with proper extraction of products"""
        query_lower = user_query.lower()
        
        query_analysis = {
            'intent': 'cart_action',
            'action': None,
            'demographics_filter': False,
            'cart_action': True,
            'query_type': 'cart',
            'extracted_info': {
                'positions': [],
                'product_names': []
            }
        }
        
        # Determine cart action
        if 'show cart' in query_lower or 'view cart' in query_lower:
            query_analysis['action'] = 'show_cart'
        elif 'clear cart' in query_lower:
            query_analysis['action'] = 'clear_cart'
        elif 'remove from cart' in query_lower or 'remove' in query_lower:
            query_analysis['action'] = 'remove_from_cart'
        elif 'add' in query_lower:
            query_analysis['action'] = 'add_to_cart'
            
            # FIXED: Extract product positions and names
            positions, product_names = self._extract_products_from_query(user_query)
            query_analysis['extracted_info']['positions'] = positions
            query_analysis['extracted_info']['product_names'] = product_names
            
            logger.info(f"CART ACTION PARSED - Positions: {positions}, Names: {product_names}")
        
        return query_analysis
    
    def _extract_products_from_query(self, query):
        """FIXED: Extract product positions and names from query"""
        positions = []
        product_names = []
        
        # Extract numbers/positions
        # Pattern 1: "add product 1,2,3 to cart"
        position_pattern1 = r'(?:product|item)s?\s+([0-9,\s]+)'
        matches1 = re.findall(position_pattern1, query, re.IGNORECASE)
        for match in matches1:
            numbers = re.findall(r'\d+', match)
            positions.extend([int(num) for num in numbers])
        
        # Pattern 2: "add 1, 2, 3 to cart" or "add first, second to cart"
        position_pattern2 = r'add\s+([0-9,\s]+|first|second|third|fourth|fifth|last)'
        matches2 = re.findall(position_pattern2, query, re.IGNORECASE)
        for match in matches2:
            if match.lower() in ['first']:
                positions.append(1)
            elif match.lower() in ['second']:
                positions.append(2)
            elif match.lower() in ['third']:
                positions.append(3)
            elif match.lower() in ['fourth']:
                positions.append(4)
            elif match.lower() in ['fifth']:
                positions.append(5)
            else:
                numbers = re.findall(r'\d+', match)
                positions.extend([int(num) for num in numbers])
        
        # Pattern 3: Direct numbers without "product" keyword
        if not positions:
            direct_numbers = re.findall(r'\b(\d+)\b', query)
            positions.extend([int(num) for num in direct_numbers if 1 <= int(num) <= 50])
        
        # Extract product names
        # Pattern: quoted strings, or specific product mentions
        quoted_pattern = r'"([^"]+)"'
        quoted_matches = re.findall(quoted_pattern, query)
        product_names.extend(quoted_matches)
        
        # Pattern: "add {product name} to cart"
        if not positions and not product_names:
            # Try to extract everything between "add" and "to cart" as potential product name
            name_pattern = r'add\s+(.+?)\s+to\s+cart'
            name_match = re.search(name_pattern, query, re.IGNORECASE)
            if name_match:
                potential_name = name_match.group(1).strip()
                # Filter out common position words
                if potential_name and not re.match(r'^(product|item|the)?\s*\d+', potential_name, re.IGNORECASE):
                    product_names.append(potential_name)
        
        return positions, product_names
    
    def _convert_llm_analysis(self, llm_result, user_query, user_id):
        """Convert LLM analysis result to expected format with FIXED cart handling"""
        query_analysis = {
            'intent': 'product_recommendation',
            'action': None,
            'demographics_filter': False,
            'cart_action': False,
            'query_type': 'general',
            'extracted_info': {}
        }
        
        intent = llm_result.get('intent', '').upper()
        
        if intent == 'CART_ACTION':
            query_analysis['intent'] = 'cart_action'
            query_analysis['cart_action'] = True
            query_analysis['action'] = llm_result.get('action', 'show_cart')
            
            product_refs = llm_result.get('product_references', [])
            product_names = llm_result.get('product_names', [])
            
            # FIXED: Ensure positions are properly extracted
            positions = self._extract_numbers_from_list(product_refs)
            if not positions and not product_names:
                # Fallback extraction
                positions, product_names = self._extract_products_from_query(user_query)
            
            query_analysis['extracted_info']['positions'] = positions
            query_analysis['extracted_info']['product_names'] = product_names
            
        elif intent == 'DEMOGRAPHIC_RECOMMENDATION':
            query_analysis['intent'] = 'demographic_recommendation'
            query_analysis['demographics_filter'] = True
            query_analysis['query_type'] = 'demographic'
            
            demo_filters = llm_result.get('demographic_filters', {})
            
            # Enhanced location handling
            if demo_filters.get('location') == 'user_location' or 'around me' in user_query.lower():
                user_location = get_user_location_from_session_or_db(user_id)
                if user_location:
                    demo_filters['location'] = user_location
                    logger.info(f"Using user location for demographic query: {user_location}")
                else:
                    demo_filters['location'] = None
                    logger.warning(f"User location not found for demographic query")
            
            # Handle age-based queries
            if 'my age' in user_query.lower() or 'age group' in user_query.lower():
                try:
                    user = users_collection.find_one({'_id': ObjectId(user_id)})
                    if user and user.get('age'):
                        user_age = user['age']
                        demo_filters['age_range'] = [max(18, user_age - 5), user_age + 5]
                except:
                    demo_filters['age_range'] = None
            
            query_analysis['extracted_info']['demographic_filters'] = demo_filters
            
        else:
            query_analysis['intent'] = 'product_recommendation'
            query_analysis['query_type'] = 'product_search'
            
            category = llm_result.get('category')
            if category:
                query_analysis['extracted_info']['category'] = category
        
        return query_analysis
    
    def _extract_numbers_from_list(self, ref_list):
        """Extract numbers from a list of references"""
        numbers = []
        
        for ref in ref_list:
            if isinstance(ref, (int, float)):
                numbers.append(int(ref))
            elif isinstance(ref, str):
                number_matches = re.findall(r'\d+', ref)
                for match in number_matches:
                    numbers.append(int(match))
        
        return numbers
    
    def _fallback_analysis(self, user_query, user_id):
        """Enhanced fallback analysis with events support and FIXED cart parsing"""
        query_lower = user_query.lower()
        
        query_analysis = {
            'intent': 'product_recommendation',
            'action': None,
            'demographics_filter': False,
            'cart_action': False,
            'query_type': 'general',
            'extracted_info': {}
        }
        
        # NEW: Add events/offers detection
        events_keywords = ['events', 'offers', 'discounts', 'sales', 'deals', 'promotions', 
                          'chaturthi', 'ganesh', 'festival', 'celebration', 'special offer', 'Halloween', 'trick or treat']
        
        if any(keyword in query_lower for keyword in events_keywords):
            query_analysis['intent'] = 'events_query'
            query_analysis['query_type'] = 'events'
            return query_analysis
        
        # FIXED: Enhanced cart action detection with product extraction
        if any(keyword in query_lower for keyword in ['add to cart', 'cart', 'remove from cart']):
            query_analysis['intent'] = 'cart_action'
            query_analysis['cart_action'] = True
            
            if 'show cart' in query_lower or 'view cart' in query_lower:
                query_analysis['action'] = 'show_cart'
            elif 'clear cart' in query_lower:
                query_analysis['action'] = 'clear_cart'
            elif 'add' in query_lower:
                query_analysis['action'] = 'add_to_cart'
                # FIXED: Extract products in fallback
                positions, product_names = self._extract_products_from_query(user_query)
                query_analysis['extracted_info']['positions'] = positions
                query_analysis['extracted_info']['product_names'] = product_names
            elif 'remove' in query_lower:
                query_analysis['action'] = 'remove_from_cart'
                # FIXED: Extract products for removal too
                positions, product_names = self._extract_products_from_query(user_query)
                query_analysis['extracted_info']['positions'] = positions
                query_analysis['extracted_info']['product_names'] = product_names
                
        elif any(keyword in query_lower for keyword in ['around me', 'people in my', 'age group', 'my location']):
            query_analysis['intent'] = 'demographic_recommendation'
            query_analysis['demographics_filter'] = True
            query_analysis['query_type'] = 'demographic'
            
            # Get user location for fallback
            user_location = get_user_location_from_session_or_db(user_id)
            query_analysis['extracted_info']['demographic_filters'] = {
                'location': user_location,
                'age_range': None,
                'gender': None
            }
                
        return query_analysis

# Create enhanced query analyzer instance with FIXED cart parsing
query_analyzer = EnhancedQueryAnalyzer()\

# =================== LLM PROCESSING ===================

ENHANCED_PRODUCT_RECOMMENDATION_PROMPT = PromptTemplate(
    input_variables=["user_query", "similar_products", "user_preferences", "demographic_context"],
    template="""You are an AI shopping assistant with deep understanding of user preferences. Recommend products that perfectly match the user's query while considering their interaction history.

User Query: {user_query}

Available Products:
{similar_products}

User Preference Analysis:
{user_preferences}

Demographic Context:
{demographic_context}

Instructions:
1. Analyze the user's query intent and product needs
2. PRIORITIZE products similar to those they've liked or added to cart
3. COMPLETELY AVOID products similar to those they've disliked
4. Factor in their preferred categories, brands, and price ranges
5. Consider demographic relevance if provided
6. Select the 6 most relevant products that match the query
7. Craft clear, concise reasons (max 12 words) that reflect product strengths aligned with user intent — highlight brand trust, seasonal suitability, flavor, value, use case, or specific benefit. Avoid generic or repetitive phrases.
   Write crisp, tailored reasons (max 12 words) that highlight a product’s specific benefit, brand appeal, seasonal fit, or unique value — clearly aligned with the user’s intent or behavior. Avoid vague, repetitive, or generic phrasing.
Return JSON format:
[
  {{
    "product_id": "product_id_here",
    "relevance_score": 0.95,
    "reason": "Heals cracked lips overnight without sticky residue"
  }}
]"""
)

# =================== ENHANCED DEMOGRAPHIC LLM PROCESSING ===================

DEMOGRAPHIC_RECOMMENDATION_PROMPT = PromptTemplate(
    input_variables=["user_query", "demographic_products", "user_preferences", "demographic_context"],
    template="""You are an AI shopping assistant analyzing demographic shopping patterns. Based on the user's location-specific query, recommend products that are popular among people in their area while considering their personal preferences.

User Query: {user_query}

Available Demographic Products (Popular in user's location):
{demographic_products}

User Preference Analysis:
{user_preferences}

Demographic Context:
{demographic_context}

Instructions:
1. Analyze the user's demographic query (e.g., "what are people around me buying?")
2. Select products that are genuinely popular in the user's location
3. PRIORITIZE products that match the user's personal preferences if they exist
4. AVOID products similar to those the user has disliked
5. Consider the local popularity and demographic appeal
6. Select the 6 most relevant demographic products that answer their query
7. Provide reasons explaining why these products are popular in their area

Return ONLY a valid JSON array in this exact format:
[
  {{
    "product_id": "product_id_here",
    "relevance_score": 0.95,
    "reason": "Popular in your area - trending electronics category"
  }}
]"""
)

def get_llm_demographic_recommendations(user_query, demographic_products, user_preferences, query_analysis, user_id):
    """Enhanced LLM-based demographic recommendations with better context"""
    try:
        demographic_logger.info(f"Getting LLM demographic recommendations for query: {user_query}")
        demographic_logger.info(f"Processing {len(demographic_products)} demographic products")
        
        # Get user preference context
        preference_context = get_enhanced_user_preference_context(user_id, user_query)
        
        # Format demographic products for LLM
        products_text = ""
        for i, product in enumerate(demographic_products[:20], 1):
            preference_indicator = ""
            product_id = str(product.get('_id', ''))
            
            # Add preference indicators
            if product_id in user_preferences.get('liked_products', []):
                preference_indicator = " [USER LIKED THIS TYPE]"
            elif product_id in user_preferences.get('cart_products', []):
                preference_indicator = " [USER ADDED SIMILAR TO CART]" 
            elif product_id in user_preferences.get('disliked_products', []):
                preference_indicator = " [USER DISLIKED - AVOID]"
            elif product.get('sub_category') in user_preferences.get('selected_subcategories', []):
                preference_indicator = f" [MATCHES USER INTEREST: {product.get('sub_category')}]"
            
            # Add location popularity info
            location_popularity = product.get('location_popularity', 1)
            popularity_indicator = f" [POPULAR: {location_popularity} people in area bought this]"
                
            products_text += f"{i}. {product.get('name', 'Unknown')} - {product.get('price', '$0.00')} ({product.get('category', '')}) - SubCategory: {product.get('sub_category', 'N/A')} - Brand: {product.get('brand', 'N/A')} - ID: {product_id}{popularity_indicator}{preference_indicator}\n"
        
        # Get demographic context
        demographic_filters = query_analysis.get('extracted_info', {}).get('demographic_filters', {})
        location_name = demographic_filters.get('location', 'your area')
        age_range = demographic_filters.get('age_range')
        gender = demographic_filters.get('gender')
        
        demographic_context = f"User's location: {location_name}"
        if age_range:
            demographic_context += f", Age range interested in: {age_range[0]}-{age_range[1]}"
        if gender:
            demographic_context += f", Gender: {gender}"
        
        # Use demographic-specific prompt template
        chain = DEMOGRAPHIC_RECOMMENDATION_PROMPT | llm_model | StrOutputParser()
        
        demographic_logger.info(f"Sending {len(demographic_products)} products to LLM for demographic analysis")
        
        response = chain.invoke({
            "user_query": user_query,
            "demographic_products": products_text,
            "user_preferences": preference_context,
            "demographic_context": demographic_context
        })
        
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            recommendations = json.loads(cleaned_response.strip())
            demographic_logger.info(f"LLM returned {len(recommendations)} demographic recommendations")
            
            return recommendations
            
        except json.JSONDecodeError as e:
            demographic_logger.error(f"Failed to parse LLM demographic response: {e}")
            demographic_logger.error(f"Raw response: {response}")
            return []
        
    except Exception as e:
        demographic_logger.error(f"Error getting LLM demographic recommendations: {e}")
        return []

def get_llm_recommendations(user_query, similar_products, user_preferences, query_analysis, user_id):
    """Enhanced LLM-based product recommendations with better personalization"""
    try:
        logger.info(f"Getting LLM recommendations for query: {user_query}")
        
        # Get user preference context with historical data
        preference_context = get_enhanced_user_preference_context(user_id, user_query)
        
        products_text = ""
        for i, product in enumerate(similar_products[:20], 1):
            preference_indicator = ""
            product_id = str(product.get('_id', ''))
            
            # Add preference indicators
            if product_id in user_preferences.get('liked_products', []):
                preference_indicator = " [USER LIKED THIS TYPE]"
            elif product_id in user_preferences.get('cart_products', []):
                preference_indicator = " [USER ADDED SIMILAR TO CART]" 
            elif product_id in user_preferences.get('disliked_products', []):
                preference_indicator = " [USER DISLIKED - AVOID]"
            elif product.get('sub_category') in user_preferences.get('selected_subcategories', []):
                preference_indicator = f" [MATCHES USER INTEREST: {product.get('sub_category')}]"
                
            products_text += f"{i}. {product.get('name', 'Unknown')} - {product.get('price', '$0.00')} ({product.get('category', '')}) - SubCategory: {product.get('sub_category', 'N/A')} - Brand: {product.get('brand', 'N/A')} - ID: {product_id}{preference_indicator}\n"
        
        chain = ENHANCED_PRODUCT_RECOMMENDATION_PROMPT | llm_model | StrOutputParser()
        
        response = chain.invoke({
            "user_query": user_query,
            "similar_products": products_text,
            "user_preferences": preference_context,
            "demographic_context": "No demographic filtering applied"
        })
        
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            recommendations = json.loads(cleaned_response.strip())
            logger.info(f"LLM returned {len(recommendations)} recommendations")
            
            return recommendations
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return []
        
    except Exception as e:
        logger.error(f"Error getting LLM recommendations: {e}")
        return []

def get_enhanced_user_preference_context(user_id, query, limit=20):
    """Get enhanced user preference context from HISTORICAL interactions"""
    try:
        # Get historical interactions
        interactions_history = get_user_interactions_history(user_id, days=30)
        
        # Get current preferences
        prefs = get_user_preferences(user_id)
        
        preferences_text = "User's preference analysis (Historical + Current):\n\n"
        
        # Show current state
        liked_products = prefs.get('liked_products', [])
        cart_products = prefs.get('cart_products', [])
        disliked_products = prefs.get('disliked_products', [])
        selected_subcategories = prefs.get('selected_subcategories', [])
        
        preferences_text += f"CURRENT STATE: {len(liked_products)} liked, {len(cart_products)} cart, {len(disliked_products)} disliked\n"
        preferences_text += f"HISTORICAL INTERACTIONS: {len(interactions_history)} interactions in last 30 days\n\n"
        
        if selected_subcategories:
            preferences_text += f"SELECTED INTERESTS: {', '.join(selected_subcategories)}\n\n"
        
        # Show preference scores
        preference_scores = prefs.get('preference_scores', {})
        if preference_scores:
            category_scores = preference_scores.get('categories', {})
            subcategory_scores = preference_scores.get('subcategories', {})
            
            if category_scores:
                top_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                preferences_text += f"TOP CATEGORY PREFERENCES: {', '.join([f'{cat} ({score:.1f})' for cat, score in top_categories])}\n"
            
            if subcategory_scores:
                top_subcategories = sorted(subcategory_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                preferences_text += f"TOP SUBCATEGORY PREFERENCES: {', '.join([f'{sub} ({score:.1f})' for sub, score in top_subcategories])}\n\n"
        
        # Show recent liked products
        if liked_products:
            preferences_text += "RECENT LIKED PRODUCTS:\n"
            for product_id in liked_products[-5:]:
                try:
                    product = products_collection.find_one({'_id': ensure_objectid(product_id)})
                    if product:
                        preferences_text += f"• {product.get('name', 'Unknown')} ({product.get('category', 'Unknown')})\n"
                except:
                    pass
            preferences_text += "\n"
        
        # Show disliked products to avoid
        if disliked_products:
            preferences_text += "DISLIKED PRODUCTS (avoid similar):\n"
            for product_id in disliked_products[-3:]:
                try:
                    product = products_collection.find_one({'_id': ensure_objectid(product_id)})
                    if product:
                        preferences_text += f"• {product.get('name', 'Unknown')} ({product.get('category', 'Unknown')})\n"
                except:
                    pass
        
        return preferences_text
        
    except Exception as e:
        preferences_logger.error(f"Error getting enhanced user preference context: {e}")
        return "No preferences available"

#====================Other users recommendation Prompt===========================
OTHER_USERS_RECOMMENDATION_PROMPT = PromptTemplate(
    input_variables=["user_query", "other_users_products", "user_preferences", "other_users_context"],
    template="""You are an AI shopping assistant analyzing collaborative filtering patterns. Based on similar users' interaction data, recommend products that match the current user's query while leveraging social proof and community preferences.

User Query: {user_query}

Products Popular Among Similar Users:
{other_users_products}

Current User's Preference Analysis:
{user_preferences}

Other Users Context:
{other_users_context}

Instructions:
1. Analyze why these products are popular among users with similar behavior patterns
2. PRIORITIZE products that align with the current user's known preferences
3. AVOID products similar to those the user has disliked
4. Consider the social proof aspect - why other users found these products appealing
5. Factor in interaction intensity (users who liked vs just viewed)
6. Select the 6 most relevant products that leverage community wisdom
7. Provide clear, engaging reasons (max 12 words) that emphasize social proof, community popularity, trending appeal, or "others like you bought this" messaging. Focus on social validation and peer behavior insights.

Return ONLY a valid JSON array in this exact format:
[
  {{
    "product_id": "product_id_here",
    "relevance_score": 0.92,
    "reason": "Popular choice among users with similar shopping patterns"
  }}
]"""
)

# Add this new function after get_llm_demographic_recommendations (around line 650)

def get_llm_other_users_recommendations(user_query, other_users_products, user_preferences, other_users_context, user_id):
    """Enhanced LLM-based other users recommendations with social proof context"""
    try:
        other_users_logger.info(f"Getting LLM other users recommendations for query: {user_query}")
        other_users_logger.info(f"Processing {len(other_users_products)} other users products")
        
        # Get user preference context
        preference_context = get_enhanced_user_preference_context(user_id, user_query)
        
        # Format other users products for LLM with social proof information
        products_text = ""
        for i, product in enumerate(other_users_products[:20], 1):
            preference_indicator = ""
            product_id = str(product.get('_id', ''))
            
            # Add preference indicators
            if product_id in user_preferences.get('liked_products', []):
                preference_indicator = " [USER LIKED THIS TYPE]"
            elif product_id in user_preferences.get('cart_products', []):
                preference_indicator = " [USER ADDED SIMILAR TO CART]" 
            elif product_id in user_preferences.get('disliked_products', []):
                preference_indicator = " [USER DISLIKED - AVOID]"
            elif product.get('sub_category') in user_preferences.get('selected_subcategories', []):
                preference_indicator = f" [MATCHES USER INTEREST: {product.get('sub_category')}]"
            
            # Add social proof indicators
            interaction_count = product.get('interaction_count', 0)
            users_count = product.get('users_count', 0)
            positive_interactions = product.get('positive_interactions', 0)
            
            social_proof_indicator = f" [SOCIAL PROOF: {positive_interactions} positive interactions by {users_count} similar users]"
                
            products_text += f"{i}. {product.get('name', 'Unknown')} - {product.get('price', '$0.00')} ({product.get('category', '')}) - SubCategory: {product.get('sub_category', 'N/A')} - Brand: {product.get('brand', 'N/A')} - ID: {product_id}{social_proof_indicator}{preference_indicator}\n"
        
        # Create other users context description
        other_users_context_text = other_users_context or "Based on users with similar interaction patterns and preferences"
        
        # Use other users specific prompt template
        chain = OTHER_USERS_RECOMMENDATION_PROMPT | llm_model | StrOutputParser()
        
        other_users_logger.info(f"Sending {len(other_users_products)} products to LLM for other users analysis")
        
        response = chain.invoke({
            "user_query": user_query,
            "other_users_products": products_text,
            "user_preferences": preference_context,
            "other_users_context": other_users_context_text
        })
        
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            recommendations = json.loads(cleaned_response.strip())
            other_users_logger.info(f"LLM returned {len(recommendations)} other users recommendations")
            
            return recommendations
            
        except json.JSONDecodeError as e:
            other_users_logger.error(f"Failed to parse LLM other users response: {e}")
            other_users_logger.error(f"Raw response: {response}")
            return []
        
    except Exception as e:
        other_users_logger.error(f"Error getting LLM other users recommendations: {e}")
        return []

# =================== MAIN PROCESSING FUNCTIONS ===================

def process_product_recommendation(user_query, user_id, query_analysis):
    """Enhanced process product recommendation request with 3 sections - FIXED with LLM processing for ALL sections"""
    try:
        # Get user preferences for reranking
        user_preferences = get_user_preferences(user_id)
        selected_subcategories = user_preferences.get('selected_subcategories', [])
        
        logger.info(f"Processing recommendation with subcategories: {selected_subcategories}")
        
        # FIXED: Handle demographic queries with LLM processing
        if query_analysis.get('demographics_filter'):
            demographic_filters = query_analysis.get('extracted_info', {}).get('demographic_filters', {})
            
            # Use ENHANCED demographic function to get raw products
            raw_demographic_products, related_products = get_enhanced_demographic_products(user_id, demographic_filters)
            
            if not raw_demographic_products:
                # If no demographic products found, return empty result
                return {
                    'success': False,
                    'message': 'No demographic data found for your location',
                    'personalized_products': [],
                    'static_products': [],
                    'other_users_products': [],
                    'query_type': 'demographic_recommendation'
                }
            
            # FIXED: Send demographic products to LLM for processing
            demographic_logger.info(f"SENDING {len(raw_demographic_products)} demographic products to LLM")
            
            llm_demographic_recommendations = get_llm_demographic_recommendations(
                user_query, raw_demographic_products, user_preferences, query_analysis, user_id
            )
            
            # Build LLM-recommended products list
            llm_recommended_products = []
            for rec in llm_demographic_recommendations[:6]:
                product_id = rec.get('product_id')
                for product in raw_demographic_products:
                    if str(product.get('_id')) == str(product_id):
                        product['reason'] = rec.get('reason', 'Popular in your area')
                        product['relevance_score'] = rec.get('relevance_score', 0.9)
                        product['llm_processed'] = True
                        llm_recommended_products.append(product)
                        break
            
            # Get remaining products for static section (not processed by LLM)
            llm_recommended_ids = {str(p.get('_id')) for p in llm_recommended_products}
            remaining_demographic_products = [
                p for p in raw_demographic_products 
                if str(p.get('_id')) not in llm_recommended_ids
            ][:8]
            
            # Store current search results for cart commands
            global current_search_results
            all_products = llm_recommended_products + remaining_demographic_products + related_products
            current_search_results[user_id] = serialize_objectid(all_products)
            
            location_name = demographic_filters.get('location', 'your area')
            if location_name == 'your area' or not location_name:
                try:
                    user = users_collection.find_one({'_id': ObjectId(user_id)})
                    if user and user.get('location'):
                        location_name = user['location']
                except:
                    location_name = 'your area'
            
            age_range = demographic_filters.get('age_range')
            gender = demographic_filters.get('gender')
            
            demographic_title = f"Popular in {location_name}"
            if age_range:
                demographic_title += f" (Ages {age_range[0]}-{age_range[1]})"
            if gender:
                demographic_title += f" ({gender})"
            
            demographic_logger.info(f"FIXED DEMOGRAPHIC RESULT - LLM processed {len(llm_recommended_products)} products")
            
            return {
                'success': True,
                'personalized_products': serialize_objectid([format_product(p) for p in llm_recommended_products]),
                'static_products': serialize_objectid([format_product(p) for p in remaining_demographic_products]),
                'other_users_products': serialize_objectid([format_product(p) for p in related_products]),
                'query_type': 'demographic_recommendation',
                'message': f"Products popular among people {demographic_title} (LLM processed)",
                'demographic_info': {
                    'location': location_name,
                    'age_range': age_range,
                    'gender': gender,
                    'llm_processed_count': len(llm_recommended_products),
                    'demographic_count': len(remaining_demographic_products),
                    'related_count': len(related_products)
                }
            }
        
        # Handle empty query - load all products with subcategory reranking
        elif (not user_query.strip() or 
            user_query.lower() in ['browse', 'show products', 'start shopping', ''] or 
            len(user_query.strip()) == 0):
            
            logger.info("Loading ALL products with subcategory reranking (empty/browse query)")
            
            all_products = get_all_products_for_reranking(user_preferences, limit=250)
            
            if not all_products:
                return {
                    'success': False,
                    'message': 'No products found',
                    'personalized_products': [],
                    'static_products': [],
                    'other_users_products': []
                }
            
            # Apply subcategory-based reranking
            reranked_products = rerank_products_by_preferences(all_products, user_preferences)
            
            static_products = reranked_products
            
            current_search_results[user_id] = serialize_objectid(static_products)
            
            return {
                'success': True,
                'personalized_products': [],
                'static_products': serialize_objectid([format_product(p) for p in static_products]),
                'other_users_products': [],
                'query_type': 'subcategory_reranking',
                'message': f"All {len(static_products)} products" + (f" reranked by your interests: {', '.join(selected_subcategories)}" if selected_subcategories else " in default order"),
                'subcategory_reranked': bool(selected_subcategories)
            }
        
        else:
            # Regular search with ENHANCED other users function WITH LLM PROCESSING
            similar_products = vector_similarity_search(user_query, limit=25)
            if not similar_products:
                return {
                    'success': False,
                    'message': 'No products found matching your query', 
                    'personalized_products': [],
                    'static_products': [],
                    'other_users_products': []
                }
            
            # Apply preference-based reranking
            reranked_products = rerank_products_by_preferences(similar_products, user_preferences)
            
            # Get LLM recommendations for personalized section
            llm_recommendations = get_llm_recommendations(
                user_query, reranked_products, user_preferences, query_analysis, user_id
            )
            
            # Build recommended products list
            recommended_products = []
            for rec in llm_recommendations[:6]:
                product_id = rec.get('product_id')
                for product in reranked_products:
                    if str(product.get('_id')) == str(product_id):
                        product['reason'] = rec.get('reason', 'Recommended for you')
                        product['relevance_score'] = rec.get('relevance_score', 0.8)
                        
                        if str(product_id) in user_preferences.get('liked_products', []):
                            product['reason'] = f" {product['reason']}"
                        elif str(product_id) in user_preferences.get('cart_products', []):
                            product['reason'] = f" {product['reason']}"
                        elif product.get('sub_category') in selected_subcategories:
                            product['reason'] = f" {product['reason']}"
                        
                        recommended_products.append(product)
                        break
            
            # Get remaining products for static section
            recommended_ids = {str(p.get('_id')) for p in recommended_products}
            static_products = [p for p in reranked_products 
                              if str(p.get('_id')) not in recommended_ids][:8]
            
            # ENHANCED: Get OTHER USERS products with LLM processing
            query_category = query_analysis.get('extracted_info', {}).get('category')
            if not query_category:
                categories = [p.get('category') for p in reranked_products[:5] if p.get('category')]
                query_category = categories[0] if categories else None
            
            # Get raw other users products
            raw_other_users_products = get_other_users_interactions(
                user_id, 
                query_category=query_category, 
                limit=15  # Get more for LLM to choose from
            )
            
            # NEW: Process other users products through LLM
            other_users_products = []
            if raw_other_users_products:
                other_users_logger.info(f"SENDING {len(raw_other_users_products)} other users products to LLM")
                
                # Create context for other users
                other_users_context = f"Products popular among users with similar preferences in {query_category or 'similar categories'}"
                
                llm_other_users_recommendations = get_llm_other_users_recommendations(
                    user_query, raw_other_users_products, user_preferences, other_users_context, user_id
                )
                
                # Build LLM-processed other users products list
                for rec in llm_other_users_recommendations[:8]:  # Limit to 8 products
                    product_id = rec.get('product_id')
                    for product in raw_other_users_products:
                        if str(product.get('_id')) == str(product_id):
                            product['reason'] = rec.get('reason', 'Popular among similar users')
                            product['relevance_score'] = rec.get('relevance_score', 0.85)
                            product['llm_processed'] = True
                            other_users_products.append(product)
                            break
                
                other_users_logger.info(f"LLM processed {len(other_users_products)} other users products")
            
            # If LLM processing failed or returned no results, fallback to raw products
            if not other_users_products and raw_other_users_products:
                other_users_logger.warning("LLM processing failed, using raw other users products")
                other_users_products = raw_other_users_products[:8]
                # Add basic reasons for fallback
                for product in other_users_products:
                    if not product.get('reason'):
                        users_count = product.get('users_count', 1)
                        product['reason'] = f"Liked by {users_count} similar users"
            
            # Store current search results for cart commands
            all_search_products = recommended_products + static_products + other_users_products
            current_search_results[user_id] = serialize_objectid(all_search_products)
            
            return {
                'success': True,
                'personalized_products': serialize_objectid([format_product(p) for p in recommended_products]),
                'static_products': serialize_objectid([format_product(p) for p in static_products]),
                'other_users_products': serialize_objectid([format_product(p) for p in other_users_products]),
                'query_type': query_analysis.get('query_type', 'product_search'),
                'message': query_analysis.get('message', ''),
                'user_preferences_applied': {
                    'subcategory_count': len([p for p in recommended_products if p.get('sub_category') in selected_subcategories]),
                    'liked_count': len(user_preferences.get('liked_products', [])),
                    'cart_count': len(user_preferences.get('cart_products', [])),
                    'disliked_count': len(user_preferences.get('disliked_products', []))
                },
                'other_users_stats': {
                    'total_products': len(other_users_products),
                    'category': query_category,
                    'llm_processed': len([p for p in other_users_products if p.get('llm_processed')]),
                    'raw_available': len(raw_other_users_products)
                }
            }
        
    except Exception as e:
        logger.error(f"Error processing product recommendation: {e}")
        return {
            'success': False,
            'message': 'An error occurred while processing your request',
            'personalized_products': [],
            'static_products': [],
            'other_users_products': []
        }
        
def process_cart_action(user_query, user_id, query_analysis):
    """FIXED: Process cart-related actions with better product matching"""
    try:
        action = query_analysis.get('action')
        logger.info(f"PROCESSING CART ACTION: {action} for query: '{user_query}'")
        
        if action == 'show_cart':
            cart_items = get_user_cart(user_id)
            total_price = sum([float(re.search(r'[\d.]+', item.get('price', '$0')).group()) * item.get('quantity', 1) 
                             for item in cart_items])
            
            return {
                'success': True,
                'is_chat_command': True,
                'chat_response': {
                    'type': 'success',
                    'message': f"🛒 Your cart has {len(cart_items)} items with total: ${total_price:.2f}",
                    'action': 'open_cart'
                }
            }
        
        elif action == 'clear_cart':
            success = clear_cart(user_id)
            return {
                'success': True,
                'is_chat_command': True,
                'chat_response': {
                    'type': 'success' if success else 'error',
                    'message': '✅ Cart cleared successfully!' if success else '⚠️ Failed to clear cart',
                    'action': 'refresh_cart' if success else None
                }
            }
        
        elif action == 'add_to_cart':
            positions = query_analysis.get('extracted_info', {}).get('positions', [])
            product_names = query_analysis.get('extracted_info', {}).get('product_names', [])
            
            logger.info(f"ADD TO CART - Positions: {positions}, Names: {product_names}")
            
            # FIXED: Get current search results properly
            current_products = current_search_results.get(user_id, [])
            logger.info(f"Current search results available: {len(current_products)} products")
            
            if not current_products:
                return {
                    'success': True,
                    'is_chat_command': True,
                    'chat_response': {
                        'type': 'error',
                        'message': 'No search results available. Please search for products first, then specify which ones to add to cart.'
                    }
                }
            
            added_products = []
            failed_products = []
            
            # FIXED: Add by position with better validation
            for pos in positions:
                try:
                    if 1 <= pos <= len(current_products):
                        product = current_products[pos - 1]
                        success = add_to_cart(user_id, product['_id'])
                        if success:
                            added_products.append(f"#{pos}: {product['name']}")
                            logger.info(f"Successfully added product at position {pos}: {product['name']}")
                        else:
                            failed_products.append(f"#{pos}: {product['name']}")
                    else:
                        failed_products.append(f"Position {pos} (out of range)")
                        logger.warning(f"Position {pos} is out of range. Available: 1-{len(current_products)}")
                except Exception as e:
                    logger.error(f"Error adding product at position {pos}: {e}")
                    failed_products.append(f"Position {pos} (error)")
            
            # FIXED: Add by name with fuzzy matching
            for name in product_names:
                try:
                    found = False
                    name_lower = name.lower().strip()
                    logger.info(f"Searching for product by name: '{name}'")
                    
                    for product in current_products:
                        product_name = product.get('name', '').lower()
                        # FIXED: Better name matching
                        if (name_lower in product_name or 
                            product_name in name_lower or
                            _fuzzy_match_name(name_lower, product_name)):
                            
                            success = add_to_cart(user_id, product['_id'])
                            if success:
                                added_products.append(f"'{product['name']}'")
                                logger.info(f"Successfully added product by name: {product['name']}")
                                found = True
                                break
                            else:
                                failed_products.append(f"'{product['name']}' (failed to add)")
                                found = True
                                break
                    
                    if not found:
                        failed_products.append(f"'{name}' (not found)")
                        logger.warning(f"Product '{name}' not found in current search results")
                        
                except Exception as e:
                    logger.error(f"Error adding product by name '{name}': {e}")
                    failed_products.append(f"'{name}' (error)")
            
            # FIXED: Provide detailed feedback
            if added_products:
                message = f"✅ Added to cart: {', '.join(added_products)}"
                if failed_products:
                    message += f"\n❌ Failed to add: {', '.join(failed_products)}"
                
                return {
                    'success': True,
                    'is_chat_command': True,
                    'chat_response': {
                        'type': 'success',
                        'message': message,
                        'action': 'refresh_cart'
                    }
                }
            elif failed_products:
                return {
                    'success': True,
                    'is_chat_command': True,
                    'chat_response': {
                        'type': 'error',
                        'message': f"❌ Failed to add: {', '.join(failed_products)}. Please check the product positions or names."
                    }
                }
            else:
                return {
                    'success': True,
                    'is_chat_command': True,
                    'chat_response': {
                        'type': 'error',
                        'message': 'No products specified to add. Please specify positions (e.g., "add 1,2,3") or product names.'
                    }
                }
        
        elif action == 'remove_from_cart':
            # FIXED: Handle remove from cart with product extraction
            positions = query_analysis.get('extracted_info', {}).get('positions', [])
            product_names = query_analysis.get('extracted_info', {}).get('product_names', [])
            
            if positions or product_names:
                cart_items = get_user_cart(user_id)
                removed_products = []
                
                # Remove by position in cart
                for pos in positions:
                    if 1 <= pos <= len(cart_items):
                        cart_item = cart_items[pos - 1]
                        success = remove_from_cart(user_id, cart_item['product_id'])
                        if success:
                            removed_products.append(f"#{pos}: {cart_item['name']}")
                
                # Remove by name
                for name in product_names:
                    for cart_item in cart_items:
                        if name.lower() in cart_item['name'].lower():
                            success = remove_from_cart(user_id, cart_item['product_id'])
                            if success:
                                removed_products.append(f"'{cart_item['name']}'")
                            break
                
                if removed_products:
                    return {
                        'success': True,
                        'is_chat_command': True,
                        'chat_response': {
                            'type': 'success',
                            'message': f"✅ Removed from cart: {', '.join(removed_products)}",
                            'action': 'refresh_cart'
                        }
                    }
            
            return {
                'success': True,
                'is_chat_command': True,
                'chat_response': {
                    'type': 'info',
                    'message': 'Please use the cart interface to remove specific items, or specify which items to remove.'
                }
            }
        
    except Exception as e:
        logger.error(f"Error processing cart action: {e}")
        return {
            'success': False,
            'is_chat_command': True,
            'chat_response': {
                'type': 'error',
                'message': 'An error occurred while processing your cart request'
            }
        }

def _fuzzy_match_name(query_name, product_name):
    """Simple fuzzy matching for product names"""
    try:
        # Split into words and check if most words match
        query_words = set(query_name.split())
        product_words = set(product_name.split())
        
        if not query_words or not product_words:
            return False
        
        # Calculate word overlap
        overlap = len(query_words.intersection(product_words))
        overlap_ratio = overlap / min(len(query_words), len(product_words))
        
        return overlap_ratio >= 0.5  # At least 50% word overlap
    except:
        return False

# Add the fuzzy match function to the analyzer class
EnhancedQueryAnalyzer._fuzzy_match_name = _fuzzy_match_name

def format_product(product):
    """Format product for display"""
    try:
        image_url = get_direct_image_url(product.get('image_url', ''))
        
        if not image_url:
            image_url = None
        
        formatted_product = {
            '_id': str(product.get('_id', '')),
            'name': product.get('name', ''),
            'price': product.get('price', '$0.00'),
            'image': image_url if image_url else "",
            'productUrl': product.get('product_url', ''),
            'category': product.get('category', ''),
            'sub_category': product.get('sub_category', ''),
            'description': product.get('description', ''),
            'brand': product.get('brand', ''),
            'rating': product.get('rating', round(random.uniform(3.5, 5.0), 1)),
            'review_count': product.get('review_count', random.randint(5, 50)),
            'reason': product.get('reason', ''),
            'relevance_score': product.get('relevance_score', 0.8),
            'preference_score': product.get('preference_score', 1),
            'preference_reason': product.get('preference_reason', ''),
            'rerank_score': product.get('rerank_score'),
            'interaction_score': product.get('interaction_score'),
            'rating_score': product.get('rating_score'),
            'interaction_count': product.get('interaction_count'),
            'users_count': product.get('users_count')
        }
        
        return formatted_product
        
    except Exception as e:
        logger.error(f"Error formatting product: {e}")
        return {
            '_id': str(product.get('_id', '')),
            'name': product.get('name', 'Unknown Product'),
            'price': '$0.00',
            'image': None,
            'productUrl': '',
            'category': '',
            'sub_category': '',
            'description': '',
            'brand': '',
            'rating': 4.0,
            'review_count': 0,
            'reason': '',
            'relevance_score': 0.8,
            'preference_score': 1,
            'preference_reason': ''
        }

# =================== CART FUNCTIONALITY ===================

def get_user_cart(user_id):
    """Get user's cart from MongoDB"""
    try:
        cart = cart_collection.find_one({'user_id': user_id})
        return cart.get('items', []) if cart else []
    except Exception as e:
        logger.error(f"Error getting user cart: {e}")
        return []

def add_to_cart(user_id, product_id, quantity=1):
    """Add product to cart and update interactions"""
    try:
        product_oid = ensure_objectid(product_id)
        product = products_collection.find_one({'_id': product_oid})
        if not product:
            return False
        
        cart_items = get_user_cart(user_id)
        
        for item in cart_items:
            if item['product_id'] == str(product['_id']):
                item['quantity'] += quantity
                break
        else:
            cart_items.append({
                'product_id': str(product['_id']),
                'name': product.get('name', ''),
                'price': product.get('price', '$0.00'),
                'image_url': product.get('image_url', ''),
                'category': product.get('category', ''),
                'brand': product.get('brand', ''),
                'quantity': quantity,
                'added_at': datetime.now().isoformat()
            })
        
        cart_collection.update_one(
            {'user_id': user_id},
            {'$set': {'user_id': user_id, 'items': cart_items, 'updated_at': datetime.now()}},
            upsert=True
        )
        
        # Store interaction in BOTH collections
        store_user_interaction(user_id, str(product['_id']), 'cart')
        
        return True
        
    except Exception as e:
        logger.error(f"Error adding to cart: {e}")
        return False

def remove_from_cart(user_id, product_id):
    """Remove product from cart"""
    try:
        cart_items = get_user_cart(user_id)
        cart_items = [item for item in cart_items if item['product_id'] != product_id]
        
        cart_collection.update_one(
            {'user_id': user_id},
            {'$set': {'items': cart_items, 'updated_at': datetime.now()}},
            upsert=True
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Error removing from cart: {e}")
        return False

def clear_cart(user_id):
    """Clear user's cart"""
    try:
        cart_collection.update_one(
            {'user_id': user_id},
            {'$set': {'items': [], 'updated_at': datetime.now()}},
            upsert=True
        )
        return True
    except Exception as e:
        logger.error(f"Error clearing cart: {e}")
        return False

# =================== FLASK ROUTES ===================

@app.route('/')
def login():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/register')
def register():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_post():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        location = request.form.get('location')
        birthday = request.form.get('birthday')
        gender = request.form.get('gender')
        
        if not all([name, email, password, location, birthday, gender]):
            return render_template('register.html', error="All fields are required")
        
        success, result = create_user(name, email, password, location, birthday, gender)
        
        if success:
            session['logged_in'] = True
            session['user_id'] = result
            session['user_email'] = email
            session['user_name'] = name
            session['location'] = location
            session['birthday'] = birthday
            session['gender'] = gender
            
            calculated_age = calculate_age_from_birthday(birthday)
            session['age'] = calculated_age if calculated_age else 18
            
            logger.info(f"User registered and logged in: {email}, Birthday: {birthday}, Age: {calculated_age}")
            return redirect(url_for('onboarding'))
        else:
            return render_template('register.html', error=result)
            
    except Exception as e:
        logger.error(f"Error in register: {e}")
        return render_template('register.html', error="An error occurred during registration")

@app.route('/login', methods=['POST'])
def login_post():
    try:
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            return render_template('login.html', error="Email and password are required")
        
        success, user = authenticate_user(email, password)
        
        if success and user:
            session['logged_in'] = True
            session['user_id'] = user['_id']
            session['user_email'] = user['email']
            session['user_name'] = user['name']
            session['location'] = user['location']
            session['age'] = user['age']
            session['gender'] = user['gender']
            
            if 'birthday' in user and user['birthday']:
                session['birthday'] = user['birthday'].strftime('%Y-%m-%d') if isinstance(user['birthday'], datetime) else str(user['birthday'])
            
            logger.info(f"User logged in: {email}")

            # MODIFIED: Send emails immediately on login
            try:
                email_logger.info(f"Sending immediate login emails for user {user['_id']}")
                
                # Send cart email immediately
                send_cart_email(user['_id'])
                
                # Send events email immediately  
                send_events_email(user['_id'])
                
                email_logger.info(f"Login emails sent successfully for user {user['_id']}")
                
            except Exception as e:
                email_logger.error(f"Error sending login emails for user {user['_id']}: {e}")

            return redirect(url_for('onboarding'))
        else:
            return render_template('login.html', error="Invalid email or password")
            
    except Exception as e:
        logger.error(f"Error in login: {e}")
        return render_template('login.html', error="An error occurred during login")


@app.route('/onboarding')
def onboarding():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        categories = get_categories_from_db()
        logger.info(f"Categories passed to template: {list(categories.keys())}")
    except Exception as e:
        logger.error(f"Error getting categories for onboarding: {e}")
        categories = get_fallback_categories()
        
    return render_template('onboarding.html', categories=categories)

@app.route('/save_preferences', methods=['POST'])
def save_preferences():
    try:
        if not session.get('logged_in'):
            return jsonify({'success': False, 'error': 'Not logged in'})
        
        data = request.get_json()
        user_id = session['user_id']
        
        selected_categories = data.get('selected_categories', [])
        selected_subcategories = data.get('selected_subcategories', [])
        
        prefs = get_user_preferences(user_id)
        prefs['selected_categories'] = selected_categories
        prefs['selected_subcategories'] = selected_subcategories
        prefs['onboarding_complete'] = True
        prefs['updated_at'] = datetime.now()
        prefs['last_activity'] = datetime.now()
        
        user_preferences_collection.update_one(
            {'user_id': user_id},
            {'$set': prefs},
            upsert=True
        )
        
        preferences_logger.info(f"Saved preferences for user {user_id}: categories={selected_categories}, subcategories={selected_subcategories}")
        
        return jsonify({'success': True, 'message': 'Preferences saved successfully'})
        
    except Exception as e:
        logger.error(f"Error saving preferences: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/skip_onboarding', methods=['POST'])
def skip_onboarding():
    try:
        if not session.get('logged_in'):
            return jsonify({'success': False, 'error': 'Not logged in'})
        
        user_id = session['user_id']
        current_time = datetime.now()
        
        default_preferences = create_default_preferences(user_id)
        default_preferences['onboarding_complete'] = True
        default_preferences['onboarding_skipped'] = True
        
        user_preferences_collection.update_one(
            {'user_id': user_id},
            {'$set': default_preferences},
            upsert=True
        )
        
        preferences_logger.info(f"Skipped onboarding for user {user_id}")
        return jsonify({'success': True, 'message': 'Onboarding skipped successfully'})
            
    except Exception as e:
        logger.error(f"Error skipping onboarding: {e}")
        return jsonify({'success': True, 'message': 'Proceeding to dashboard'})

@app.route('/get_dashboard_data', methods=['GET'])
def get_dashboard_data():
    """FIXED: Get dashboard sections based on user preferences and historical interactions - ENHANCED WITH EVENTS"""
    try:
        if not session.get('logged_in'):
            return jsonify({'success': False, 'error': 'Not logged in'})
        
        user_id = session['user_id']
        prefs = get_user_preferences(user_id)
        
        # Determine if user has preferences
        has_preferences = bool(prefs.get('selected_subcategories') or 
                              prefs.get('liked_products') or 
                              prefs.get('cart_products'))
        
        sections = {}

        # 🎂 NEW: Birthday Celebration Section (Highest Priority)
        birthday_celebration = get_birthday_celebration(user_id)
        if birthday_celebration:
            sections['birthday_celebration'] = {
                'title': birthday_celebration['celebration_message'],
                'subtitle': birthday_celebration['celebration_subtitle'],
                'description': f"Special birthday offers valid during your birthday week!",
                'icon': '🎂',
                'birthday_data': birthday_celebration,
                'has_more': False,
                'is_personal': True
            }
            birthday_logger.info(f"🎉 Added birthday celebration for {birthday_celebration['user_name']}")
        
        # FIXED: Current Events & Offers section with proper error handling
        try:
            current_events = get_current_events(user_id)
            logger.info(f"Dashboard data generated with {len(current_events)} active events")
            
            if current_events:
                sections['current_events'] = {
                    'title': 'Current Offers & Events',
                    'subtitle': 'Exclusive deals just for you',
                    'description': 'Check out ongoing events and personalized offers',
                    'icon': '🎉',
                    'events': current_events,
                    'has_more': False
                }
                logger.info(f"Added current_events section with {len(current_events)} events")
            else:
                logger.info("No active events found, skipping current_events section")
        except Exception as e:
            logger.error(f"Error getting current events for dashboard: {e}")
        
        # SECTION 1: Continue Shopping (historical data)
        continue_shopping = get_continue_shopping_products(user_id, limit=6)
        
        # Add offer information to continue shopping products
        if sections.get('current_events'):
            continue_shopping = [add_offer_to_product(p, sections['current_events']['events']) for p in continue_shopping]
        
        sections['continue_shopping'] = {
            'products': continue_shopping,
            'title': 'Continue Shopping',
            'subtitle': 'Based on your interaction history',
            'description': 'Products similar to ones you\'ve previously liked or added to cart',
            'icon': '🔄',
            'has_more': len(get_continue_shopping_products(user_id, limit=50)) > 6
        }
        # SECTION 2: Your Shopping Trends (only if user has preferences) - ENHANCED
        if has_preferences:
            shopping_trends = get_shopping_trends_products(user_id, limit=8)

            # Add offer information to shopping trends products
            if sections.get('current_events'):
                shopping_trends = [add_offer_to_product(p, sections['current_events']['events']) for p in shopping_trends]

            if shopping_trends:
                # Add enhanced subtitle and description logic
                analysis = analyze_user_selections(prefs)
        
                subtitle = "Personalized recommendations"
                description = "Products matching your selected interests" 

                if analysis:
                    if analysis['selection_type'] == 'subcategories_only':
                        subtitle = f"Based on your {len(analysis['subcategory_selections'])} selected interests"
                        description = f"Products from: {', '.join(analysis['subcategory_selections'][:3])}" + ("..." if len(analysis['subcategory_selections']) > 3 else "")
                    elif analysis['selection_type'] == 'categories_only':
                        subtitle = f"Based on your {len(analysis['category_selections'])} selected categories"
                        description = f"Products from: {', '.join(analysis['category_selections'])}"
                    elif analysis['selection_type'] == 'mixed':
                        subtitle = "Based on your diverse interests"
                        description = f"Mix from {len(analysis['category_selections'])} categories and {len(analysis['subcategory_selections'])} specific interests"
        
                sections['shopping_trends'] = {
                    'products': shopping_trends,
                    'title': 'Based on your Shopping trends',
                    'subtitle': subtitle,
                    'description': description,
                    'icon': '✨',
                    'has_more': False,
                    'selection_analysis': {
                        'total_products': len(shopping_trends),
                        'selection_type': analysis['selection_type'] if analysis else 'none',
                        'user_intent': analysis['user_intent'] if analysis else 'none',
                        'categories_selected': len(analysis['category_selections']) if analysis else 0,
                        'subcategories_selected': len(analysis['subcategory_selections']) if analysis else 0
            }
        }
        
        # SECTION 3: All Products (category-based)
        categories_data = get_categories_with_subcategories()
        sections['all_products'] = {
            'title': 'All Products',
            'subtitle': 'Browse by categories',
            'description': 'Explore our complete product catalog organized by categories',
            'icon': '🛍️',
            'categories': categories_data
        }
        
        # Get interaction statistics
        interactions_history = get_user_interactions_history(user_id, days=30)
        
        logger.info(f"✅ Dashboard data generated successfully")
        if sections.get('current_events'):
            logger.info(f"📅 EVENTS: {len(sections['current_events']['events'])} active events included")
        if birthday_celebration:
            birthday_logger.info(f"🎂 Birthday celebration included for user")
        
        return jsonify({
            'success': True,
            'sections': sections,
            'has_preferences': has_preferences,
            'has_birthday_celebration': birthday_celebration is not None,
            'user_state': {
                'selected_subcategories': prefs.get('selected_subcategories', []),
                'selected_categories': prefs.get('selected_categories', []),
                'liked_count': len(prefs.get('liked_products', [])),
                'cart_count': len(prefs.get('cart_products', [])),
                'interactions_count': len(interactions_history),
                'total_historical_interactions': len(interactions_history),
                'has_shopping_trends_section': 'shopping_trends' in sections
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_category_products', methods=['POST'])
def get_category_products():
    """Get products by category/subcategory"""
    try:
        if not session.get('logged_in'):
            return jsonify({'success': False, 'error': 'Not logged in'})
        
        data = request.get_json()
        category = data.get('category')
        subcategory = data.get('subcategory')
        limit = data.get('limit', 20)
        
        user_id = session['user_id']
        prefs = get_user_preferences(user_id)
        
        # Build query
        query = {}
        if category:
            query['category'] = category
        if subcategory:
            query['sub_category'] = subcategory
        
        # Get products
        products_cursor = products_collection.find(query).limit(limit * 2)
        
        products = []
        for product in products_cursor:
            product_data = format_product_for_dashboard(product, user_id, prefs)
            products.append(product_data)
        
        # Apply reranking
        reranked_products = enhanced_rerank_with_rating(products, prefs)
        
        return jsonify({
            'success': True,
            'products': reranked_products[:limit],
            'total_found': len(reranked_products),
            'category': category,
            'subcategory': subcategory
        })
        
    except Exception as e:
        logger.error(f"Error getting category products: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_continue_shopping_more', methods=['GET'])
def get_continue_shopping_more():
    """Get more products for Continue Shopping section"""
    try:
        if not session.get('logged_in'):
            return jsonify({'success': False, 'error': 'Not logged in'})
        
        user_id = session['user_id']
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 6))
        
        # Get all continue shopping products
        all_products = get_continue_shopping_products(user_id, limit=limit + offset)
        
        # Return products after offset
        more_products = all_products[offset:offset + limit] if len(all_products) > offset else []
        
        return jsonify({
            'success': True,
            'products': more_products,
            'has_more': len(all_products) > offset + limit
        })
        
    except Exception as e:
        logger.error(f"Error getting more continue shopping products: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user_name = session.get('user_name', 'User')
    
    try:
        prefs = user_preferences_collection.find_one({'user_id': user_id})
        if not prefs:
            prefs = create_default_preferences(user_id)
        else:
            if 'selected_categories' not in prefs:
                prefs['selected_categories'] = []
        
        user_state = {
            'has_preferences': bool(prefs.get('selected_subcategories') or 
                                  prefs.get('liked_products') or 
                                  prefs.get('cart_products')),
            'selected_subcategories': prefs.get('selected_subcategories', []),
            'selected_categories': prefs.get('selected_categories', []),
            'onboarding_skipped': prefs.get('onboarding_skipped', False),
            'interactions_count': len(prefs.get('recent_interactions', [])),
            'liked_count': len(prefs.get('liked_products', [])),
            'cart_count': len(prefs.get('cart_products', []))
        }
        
        logger.info(f"Enhanced Dashboard: User {user_id} has {user_state}")
        
        return render_template('chat.html', 
                             user_state=user_state,
                             user_name=user_name,
                             selected_categories=prefs.get('selected_categories', []),
                             dashboard_mode=True)
                             
    except Exception as e:
        logger.error(f"Error loading enhanced dashboard: {e}")
        return render_template('chat.html',
                             user_state={
                                 'has_preferences': False,
                                 'selected_subcategories': [],
                                 'selected_categories': [],
                                 'onboarding_skipped': True,
                                 'interactions_count': 0,
                                 'liked_count': 0,
                                 'cart_count': 0
                             },
                             user_name=user_name,
                             selected_categories=[],
                             dashboard_mode=True)

@app.route('/search', methods=['POST'])
def search():
    """Enhanced search with events, birthday queries, and FIXED email triggering capabilities."""
    try:
        if not session.get('logged_in'):
            return jsonify({'success': False, 'error': 'Not logged in'})
        
        data = request.get_json()
        query = data.get('query', '').strip()
        user_id = session['user_id']
        dashboard_mode = data.get('dashboard_mode', False)

        # --- Email Request Handling ---
        email_keywords = ['send to my email', 'send me', 'email these', 'to my email', 'send these', 'email this', 'send this']
        is_email_request = any(keyword in query.lower() for keyword in email_keywords)

        if is_email_request:
            email_logger.info(f"Email request detected: '{query}'")
            
            # FIXED: Always check for previous results first, regardless of query content
            previous_results = last_search_results.get(user_id)
            
            if previous_results:
                email_logger.info(f"Using previous search results for email (user {user_id})")
                
                user = users_collection.find_one({'_id': ObjectId(user_id)})
                if user:
                    # Get ONLY the first section products (personalized_products)
                    personalized_products = previous_results.get('personalized_products', [])
                    
                    if personalized_products:
                        # Send email with first section products only
                        send_recommendations_email(user_id, personalized_products)
                        email_logger.info(f"✅ Successfully sent {len(personalized_products)} personalized products to {user['email']}")
                        
                        # Return email confirmation response immediately - NO SEARCH PROCESSING
                        return jsonify({
                            'success': True,
                            'email_sent': True,
                            'email_message': f'Your {len(personalized_products)} personalized recommendations have been sent to your email!',
                            'products_sent': len(personalized_products),
                            'email_type': 'personalized_recommendations',
                            'user_email': user['email'],
                            'sent_at': datetime.now().isoformat()
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'email_sent': False,
                            'message': 'No personalized recommendations found from previous search to email.',
                            'error': 'No personalized products available'
                        })
                else:
                    return jsonify({
                        'success': False,
                        'email_sent': False,
                        'message': 'User not found.',
                        'error': 'User not found'
                    })
            else:
                # No previous results found - suggest user to search first
                return jsonify({
                    'success': False,
                    'email_sent': False,
                    'message': 'No previous search results found. Please search for products first, then request email.',
                    'error': 'No previous search results'
                })
        
        # FIXED: Only process normal search if it's NOT an email request
        else:
            clean_query = query
            
            # --- Dashboard Mode Shortcut ---
            if dashboard_mode and (not clean_query or clean_query.lower() in ['browse', 'start shopping', '']):
                return redirect(url_for('get_dashboard_data'))
            
            # --- Generate Results for Normal Search (No Email) ---
            logger.info(f"Enhanced Search (Dashboard: {dashboard_mode}, Email Request: False): '{clean_query}' by user {user_id}")
            
            result = {}
            if any(keyword in clean_query.lower() for keyword in ['events', 'offers', 'birthday']):
                events = get_current_events(user_id)
                personalized_message = ""
                if any(e.get('offers') and any(o.get('matches_preferences') for o in e['offers']) for e in events):
                    personalized_message = " personalized for you"
                
                result = {
                    'success': True,
                    'events': events,
                    'message': f"Here are the current events and offers{personalized_message}",
                    'query_type': 'events',
                    'is_events_response': True
                }
                # Special check for birthday
                if 'birthday' in clean_query.lower():
                    birthday_data = get_birthday_celebration(user_id)
                    if birthday_data:
                        result['query_type'] = 'birthday'
                        result['birthday_data'] = birthday_data
            else:
                query_analysis = query_analyzer.analyze_query(clean_query, user_id)

                if query_analysis['intent'] == 'cart_action':
                    result = process_cart_action(clean_query, user_id, query_analysis)
                else: # Handles 'product_recommendation' and 'demographic_recommendation'
                    result = process_product_recommendation(clean_query, user_id, query_analysis)
                
                result['dashboard_search'] = dashboard_mode

            # --- Store Results for Future Email Requests ---
            if result.get('success') and user_id:
                last_search_results[user_id] = result.copy()
                email_logger.info(f"Stored search results for user {user_id} for future email requests")

            return jsonify(result)

    except Exception as e:
        logger.error(f"Error in enhanced search: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/track_interaction', methods=['POST'])
def track_interaction():
    try:
        if not session.get('logged_in'):
            return jsonify({'success': False, 'error': 'Not logged in'})
        
        data = request.get_json()
        user_id = session['user_id']
        product_id = data.get('product_id')
        interaction_type = data.get('type')
        
        if not product_id or not interaction_type:
            return jsonify({'success': False, 'error': 'Product ID and interaction type required'})
        
        # Store in BOTH user_interactions AND user_preferences collections
        success, product_info = store_user_interaction(user_id, product_id, interaction_type)
        
        interactions_logger.info(f"Tracked {interaction_type} interaction for product {product_id} by user {user_id}")
        
        response = {'success': success}
        
        if success and product_info:
            response['product_info'] = product_info
            response['interaction_type'] = interaction_type
        
        return jsonify(response)
        
    except Exception as e:
        interactions_logger.error(f"Error tracking interaction: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/cart', methods=['GET', 'POST'])
def cart():
    try:
        if not session.get('logged_in'):
            return jsonify({'success': False, 'error': 'Not logged in'})
        
        user_id = session['user_id']
        
        if request.method == 'GET':
            cart_items = get_user_cart(user_id)
            total_price = sum([float(re.search(r'[\d.]+', item.get('price', '$0')).group()) * item.get('quantity', 1) 
                             for item in cart_items])
            
            return jsonify({
                'success': True,
                'cart': {
                    'items': cart_items,
                    'total_items': len(cart_items),
                    'total_price': round(total_price, 2)
                }
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            action = data.get('action')
            
            if action == 'add':
                product_id = data.get('product_id')
                quantity = data.get('quantity', 1)
                success = add_to_cart(user_id, product_id, quantity)
                
                if success:
                    cart_items = get_user_cart(user_id)
                    total_price = sum([float(re.search(r'[\d.]+', item.get('price', '$0')).group()) * item.get('quantity', 1) 
                                     for item in cart_items])
                    
                    return jsonify({
                        'success': True,
                        'message': 'Product added to cart',
                        'cart': {
                            'items': cart_items,
                            'total_items': len(cart_items),
                            'total_price': round(total_price, 2)
                        }
                    })
                else:
                    return jsonify({'success': False, 'error': 'Failed to add to cart'})
            
            elif action == 'remove':
                product_id = data.get('product_id')
                success = remove_from_cart(user_id, product_id)
                
                if success:
                    cart_items = get_user_cart(user_id)
                    total_price = sum([float(re.search(r'[\d.]+', item.get('price', '$0')).group()) * item.get('quantity', 1) 
                                     for item in cart_items])
                    
                    return jsonify({
                        'success': True,
                        'message': 'Product removed from cart',
                        'cart': {
                            'items': cart_items,
                            'total_items': len(cart_items),
                            'total_price': round(total_price, 2)
                        }
                    })
                else:
                    return jsonify({'success': False, 'error': 'Failed to remove from cart'})
            
            elif action == 'clear':
                success = clear_cart(user_id)
                
                if success:
                    return jsonify({
                        'success': True,
                        'message': 'Cart cleared',
                        'cart': {
                            'items': [],
                            'total_items': 0,
                            'total_price': 0.0
                        }
                    })
                else:
                    return jsonify({'success': False, 'error': 'Failed to clear cart'})
        
    except Exception as e:
        logger.error(f"Error in cart endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart_endpoint():
    try:
        if not session.get('logged_in'):
            return jsonify({'success': False, 'error': 'Not logged in'})
        
        data = request.get_json()
        user_id = session['user_id']
        product_id = data.get('product_id')
        quantity = data.get('quantity', 1)
        
        success = add_to_cart(user_id, product_id, quantity)
        
        if success:
            cart_items = get_user_cart(user_id)
            return jsonify({
                'success': True,
                'message': 'Product added to cart',
                'cart_count': len(cart_items)
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to add to cart'})
        
    except Exception as e:
        logger.error(f"Error in add_to_cart endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_categories')
def get_categories():
    try:
        categories = get_categories_from_db()
        return jsonify({'success': True, 'categories': categories})
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_dynamic_recommendations', methods=['POST'])
def get_dynamic_recommendations():
    """Enhanced endpoint to get dynamic product recommendations based on user interactions"""
    try:
        if not session.get('logged_in'):
            return jsonify({'success': False, 'error': 'Not logged in'})
        
        user_id = session['user_id']
        data = request.get_json()
        interaction_type = data.get('interaction_type')
        product_info = data.get('product_info')
        
        if interaction_type in ['like', 'cart']:
            if product_info:
                query = f"{product_info.get('name', '')} {product_info.get('category', '')} {product_info.get('brand', '')}"
                similar_products = vector_similarity_search(query, limit=30)
                
                user_preferences = get_user_preferences(user_id)
                filtered_products = [p for p in similar_products if p.get('_id') != product_info.get('_id')]
                reranked_products = rerank_products_by_preferences(filtered_products, user_preferences)
                
                recommendations = []
                for product in reranked_products[:3]:
                    recommendations.append(format_product(product))
                
                return jsonify({
                    'success': True,
                    'recommendations': recommendations,
                    'trigger_product': product_info,
                    'interaction_type': interaction_type
                })
        
        return jsonify({'success': True, 'recommendations': []})
        
    except Exception as e:
        logger.error(f"Error getting dynamic recommendations: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/logout')
def logout():
    user_id = session.get('user_id')
    
    if user_id:
        try:
            email_logger.info(f"Sending immediate logout emails for user {user_id}")
            send_cart_email(user_id)
            send_events_email(user_id)
            email_logger.info(f"Logout emails sent successfully for user {user_id}")
            logger.info(f"User {user_id} logged out, immediate emails triggered.")
            
            # Clear stored search results for this user
            if user_id in last_search_results:
                del last_search_results[user_id]
                email_logger.info(f"Cleared stored search results for user {user_id}")
            
        except Exception as e:
            email_logger.error(f"Error during logout email trigger for user {user_id}: {e}")
    
    session.clear()
    return redirect(url_for('login'))



# =================== ERROR HANDLERS ===================

@app.errorhandler(404)
def not_found_error(error):
    if request.is_json:
        return jsonify({'success': False, 'error': 'Resource not found'}), 404
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    if request.is_json:
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    return render_template('error.html', error="Internal server error"), 500

# =================== INITIALIZATION AND MAIN ===================

if __name__ == '__main__':
    try:
        logger.info("🚀 Starting Complete Enhanced Shopping System with FIXED Events Integration and Birthday Celebrations...")
        logger.info("=" * 80)
        
        logger.info("1. Initializing Vertex AI...")
        initialize_vertex_ai()
        
        logger.info("2. Setting up Qdrant collections...")
        setup_qdrant_collections()
        
        logger.info("3. FIXING and ensuring events data...")
        ensure_events_data()  # ADDED: This is the key fix
        
        logger.info("4. Setting up COMPLETE ENHANCED SYSTEM WITH FIXED EVENTS AND BIRTHDAY CELEBRATIONS...")
        logger.info("   • FIXED events date handling and timezone issues")
        logger.info("   • FIXED events collection integration")
        logger.info("   • Enhanced user_interactions collection")
        logger.info("   • Enhanced user_preferences schema")
        logger.info("   • Historical data for continue shopping")
        logger.info("   • Enhanced other users interactions with explicit iteration")
        logger.info("   • Fixed demographic recommendations for ALL categories")
        logger.info("   • Enhanced location-based queries")
        logger.info("   • Real-time preference scoring")
        logger.info("   • Multi-session interaction tracking")
        logger.info("   • Enhanced query analysis with location detection")
        logger.info("   • Direct image URL processing")
        logger.info("   • FIXED Current events and offers dashboard section")
        logger.info("   • FIXED Event-based product offers with personalization")
        logger.info("   • FIXED Events query handling and search integration")
        logger.info("   🎂 Birthday week detection (3 days before/after)")
        logger.info("   🎁 Personalized birthday offers based on preferences")
        logger.info("   🎉 Special birthday section on dashboard")
        logger.info("   📅 Age calculation and birthday messaging")
        logger.info("   ✨ Events integration")
        logger.info("   💝 User interaction tracking")
        logger.info("   🛒 Enhanced cart functionality")
        
        logger.info("=" * 80)
        logger.info("✅ COMPLETE ENHANCED SHOPPING SYSTEM WITH FIXED EVENTS AND BIRTHDAY CELEBRATIONS READY!")
        logger.info("🔹 Key Fixes Applied:")
        logger.info("  ✅ FIXED EVENTS DATE HANDLING: Proper timezone and dateutil parsing")
        logger.info("  ✅ FIXED EVENTS INITIALIZATION: ensure_events_data() function")
        logger.info("  ✅ FIXED DATE CONVERSION: String to datetime object conversion")
        logger.info("  ✅ DUAL STORAGE: user_interactions + user_preferences")
        logger.info("  ✅ ENHANCED OTHER USERS: Explicit user iteration & filtering")
        logger.info("  ✅ FIXED DEMOGRAPHICS: ALL categories for 'around me' queries")
        logger.info("  ✅ HISTORICAL CONTINUE SHOPPING: 30+ days interaction history")
        logger.info("  ✅ ENHANCED QUERY ANALYSIS: Better location detection + events")
        logger.info("  ✅ REAL-TIME PREFERENCE SCORING: Category & subcategory weights")
        logger.info("  ✅ ENHANCED LLM CONTEXT: Historical + current preferences")
        logger.info("  ✅ CONSISTENT USER_ID: Across all collections")
        logger.info("  ✅ DIRECT IMAGE PROCESSING: No fallback dependencies")
        logger.info("  ✅ COMPREHENSIVE LOGGING: Specialized loggers for debugging")
        logger.info("  ✅ FIXED EVENT-BASED OFFERS: Product offers with discount information")
        logger.info("  ✅ FIXED PERSONALIZED EVENTS: Offer highlighting based on preferences")
        logger.info("  ✅ FIXED EVENTS SEARCH: Dedicated handling for offer/event queries")
        logger.info("🎂 New Birthday Features:")
        logger.info("  ✅ BIRTHDAY WEEK DETECTION: 7-day celebration window")
        logger.info("  ✅ PERSONALIZED OFFERS: Based on user preferences & history")
        logger.info("  ✅ SPECIAL DASHBOARD SECTION: Priority birthday section")
        logger.info("  ✅ DYNAMIC MESSAGING: Personalized birthday greetings")
        logger.info("  ✅ AGE CALCULATION: Real-time age updates")
        logger.info("  ✅ BIRTHDAY OFFERS: 15-40% discounts on preferred products")
        logger.info("  ✅ SPECIAL STYLING: Unique birthday theme design")
        logger.info("=" * 80)
        
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except Exception as e:
        logger.error(f"⚠️ Failed to start application: {e}")
        raise