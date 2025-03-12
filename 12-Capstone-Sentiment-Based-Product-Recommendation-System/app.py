from flask import Flask, render_template, request, redirect, url_for, flash
import model  # Importing the updated model module
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')


# Initializing Flask application
app = Flask(__name__)
app.secret_key = 'secret_key'  # Required for flashing messages

# Define the model_type variable
model_type = "rf_base" 

# Mapping technical model names to user-friendly names
model_names = {
    "rf_base": "Random Forest Base",
    "lgbm_base": "LightGBM Base Model"
}

# List of valid user IDs
valid_userid = [
     'joshua', 'nicole', 'samantha', 'rebecca', 'walker557', 'kimmie', 'charlie',
]

def to_camel_case(text):
    """Converts text to CamelCase format."""
    return ' '.join(word.capitalize() for word in text.split())

def format_row_data(row_data):
    """Formats row data for better readability in the UI."""
    return [[to_camel_case(str(cell)) for cell in row] for row in row_data]

@app.route('/')
def view():
    """Displays the home page where the user can input their user name."""
    return render_template("index.html", users=valid_userid)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend_top5():
    if request.method == 'GET':
        flash("Invalid request method. Please use the form to submit.", "danger")
        return redirect(url_for('view'))
    try:
        user_name = request.form['username']  # Match HTML form field
        logging.info(f"Received recommendation request for user: {user_name}")
        model_type = request.form['model_type']  # Get selected model type

        if user_name not in valid_userid:
            logging.info("Invalid user: No recommendations available.")
            raise ValueError("Invalid user: No recommendations available. Please provide a valid username.")
        
        # Validate model_type
        if model_type not in model_names.keys():
            raise ValueError(f"Invalid model type: {model_type}. Allowed models: {list(model_names.values())}")

        # Convert technical model name to user-friendly name
        selected_model_name = model_names.get(model_type, model_type)
        logging.info(f"Received recommendation request for user: {user_name} using model: {model_type}[{selected_model_name}]")
        
         # Get recommendations from model.py
        top5_products, from_cache = model.product_recommendations_user(user_name, model_type)
        logging.info(f"Extracted Top 5 products for user {user_name} (from cache: {from_cache}) using model {model_type}: {top5_products}")
        
        # Ensure the response is a list and process it properly
        if not isinstance(top5_products, list) or len(top5_products) == 0:
            logging.warning("No recommendations found or incorrect format returned.")
            return render_template('index.html', message="No recommendations found for this user.", users=valid_userid)
        
        # Process recommendations for rendering in UI
        # formatted_row_data = [[rec['name'], rec['post_sent_percentage'], rec['most_common_sentiment']] for rec in top5_products]
        formatted_row_data = [[rec['name'], rec['most_common_sentiment']] for rec in top5_products]

        logging.info(f"Formatted row data: {formatted_row_data}")

        # return render_template('index.html', row_data=formatted_row_data, column_names=['Product Name', 'Positive Sentiment %', 'Most Common Sentiment'], 
        #     users=valid_userid, user_name=user_name, from_cache=from_cache, model_type=selected_model_name)
        return render_template('index.html', row_data=formatted_row_data, column_names=['Product Name', 'Most Common Sentiment'], 
            users=valid_userid, user_name=user_name, from_cache=from_cache, model_type=selected_model_name)

    except Exception as e:
        logging.error(f"Exception occurred: {str(e)}", exc_info=True)
        flash(f"Error: Exception occurred: {str(e)}",  "danger")
        time.sleep(2)  # Wait for 2 seconds before redirecting
        return render_template('index.html', message="Error occurred during recommendation.", users=valid_userid)

@app.errorhandler(404)
def page_not_found(e):
    """Redirects all invalid routes to the home page."""
    flash(f"Request to invalid routes, redirecting to the home page.", "danger")
    return render_template("index.html", users=valid_userid), 404

if __name__ == '__main__':
    app.run(debug=True)
