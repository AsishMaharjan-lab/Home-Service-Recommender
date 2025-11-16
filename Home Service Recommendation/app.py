# app.py (Updated with Login Redirects to Welcome Page)
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import pandas as pd
import os
import sys
import joblib
import json
import datetime
import webbrowser
from werkzeug.security import generate_password_hash, check_password_hash # For password hashing

# Add the ml_logic directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_logic'))

# Import from ml_logic
from train import train_and_save_model
from recommend import load_processed_data_and_encoders, get_recommendations

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your_super_secret_key_here_change_this_!') # KEEP THIS SECRET!
# Use a timestamp for cache busting to ensure unique CSS URL on each restart
app.config['VERSION'] = str(int(datetime.datetime.now().timestamp()))
print(f"App VERSION for cache busting: {app.config['VERSION']}") # Debug print

# Global variables for ML assets
processed_data_global = None
encoders_global = None

# Local storage file paths
USERS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'users.json')
REVIEWS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'reviews.json')
BOOKINGS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'bookings.json')

# Admin credentials (for simplicity, NOT for production!)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = generate_password_hash("adminpass") # Hash the admin password

# --- Helper functions for local JSON storage ---
def load_json_data(filepath, default_value=[]):
    """Loads data from a JSON file."""
    if not os.path.exists(filepath):
        # Ensure directory exists before creating file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(default_value, f)
        return default_value
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json_data(filepath, data):
    """Saves data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True) # Ensure directory exists
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

# Global in-memory storage for users, reviews, bookings
users_data = []
reviews_data = []
bookings_data = []

def load_all_local_data():
    """Loads all local JSON data into memory on app startup."""
    global users_data, reviews_data, bookings_data
    print("Loading local user, review, and booking data...")
    users_data = load_json_data(USERS_FILE, [])
    reviews_data = load_json_data(REVIEWS_FILE, [])
    bookings_data = load_json_data(BOOKINGS_FILE, [])
    print(f"Loaded {len(users_data)} users, {len(reviews_data)} reviews, {len(bookings_data)} bookings.")


def load_ml_assets():
    """
    Loads preprocessed data and encoders from the 'models' directory.
    If not found, it triggers the training process to create them.
    This function will be called once on app startup.
    """
    global processed_data_global, encoders_global
    project_root = os.path.dirname(__file__)
    model_dir = os.path.join(project_root, 'models')
    data_path = os.path.join(project_root, 'data', 'service_dataset.csv')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    processed_df_path = os.path.join(model_dir, "processed_data.pkl")
    encoders_path = os.path.join(model_dir, "encoders.pkl")

    if not os.path.exists(processed_df_path) or not os.path.exists(encoders_path):
        print("Processed data or encoders not found. Running initial training/preprocessing.")
        if not os.path.exists(data_path):
            print(f"Error: Dataset not found at {data_path}. Please ensure 'service_dataset.csv' is in the 'data/' directory.")
            return
        train_and_save_model(data_path=data_path, model_dir=model_dir)
    else:
        print("Found existing processed data and encoders. Loading them.")

    processed_data_global, encoders_global = load_processed_data_and_encoders(model_dir=model_dir)

    if processed_data_global is None or encoders_global is None:
        print("CRITICAL ERROR: Failed to load ML assets. Application may not function correctly.")
    else:
        print("ML assets (processed data, encoders) loaded successfully for the application.")

# --- Helper function to get current user info for templates ---
@app.context_processor
def inject_user():
    """Injects user information into all templates."""
    user_id = session.get('user_id')
    is_admin = session.get('is_admin', False)
    username = None
    if user_id:
        user = next((u for u in users_data if u['id'] == user_id), None)
        if user:
            username = user.get('username')
        elif user_id == ADMIN_USERNAME: # For admin, use ADMIN_USERNAME as username
            username = ADMIN_USERNAME
    # For local storage, app_id is just a placeholder
    app_id = os.environ.get('__app_id', 'local-app-id')
    return dict(current_user_id=user_id, is_admin=is_admin, app_id=app_id, current_username=username)


# --- New Routes for Authentication ---
@app.route('/')
def root():
    """Redirects the root URL to the welcome page."""
    return redirect(url_for('welcome'))

@app.route('/welcome')
def welcome():
    """Renders the welcome page."""
    return render_template('welcome.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles user registration."""
    if request.method == 'POST':
        username = request.form.get('username') # Get username
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            flash('Username, email, and password are required!', 'danger')
            return redirect(url_for('register'))

        # Check if email already exists
        if any(user['email'] == email for user in users_data):
            flash('Email already registered!', 'danger')
            return redirect(url_for('register'))
        
        # Check if username already exists
        if any(user.get('username') == username for user in users_data):
            flash('Username already taken!', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user_id = f"user_{len(users_data) + 1}" # Simple sequential ID

        new_user = {
            'id': new_user_id,
            'username': username, # Store username
            'email': email,
            'password': hashed_password # Store hashed password (already a string)
        }
        users_data.append(new_user)
        save_json_data(USERS_FILE, users_data)

        # Removed automatic login
        # session['user_id'] = new_user_id
        # session['is_admin'] = False
        flash('Registration successful! Please log in with your new account.', 'success') # Updated flash message
        print(f"User registered: {new_user_id} ({username}). Redirecting to login.")
        return redirect(url_for('login')) # Redirect to login page
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        admin_login = request.form.get('admin_login')

        if admin_login and email == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session['user_id'] = ADMIN_USERNAME
            session['is_admin'] = True
            flash('Admin login successful!', 'success')
            print(f"Admin logged in: {ADMIN_USERNAME}")
            return redirect(url_for('welcome')) # Redirect admin to welcome page
        elif not admin_login: # Regular user login attempt
            user = next((u for u in users_data if u['email'] == email), None)
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['is_admin'] = False
                flash('Login successful!', 'success')
                print(f"User logged in: {user['id']}")
                return redirect(url_for('welcome')) # Redirect user to welcome page
            else:
                flash('Invalid email or password for user login.', 'danger')
                print(f"Failed user login attempt: email={email}")
                return redirect(url_for('login'))
        else: # Admin login attempt with wrong credentials
            flash('Invalid credentials for admin login.', 'danger')
            print(f"Failed admin login attempt: email={email}, password={password}")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    """Handles user logout."""
    session.pop('user_id', None)
    session.pop('is_admin', None)
    flash('You have been logged out.', 'info')
    print("User logged out.")
    return redirect(url_for('welcome'))

# --- Existing Routes (now with login checks) ---
@app.route('/index')
def index():
    """
    Renders the main page of the recommendation system.
    Requires user to be logged in.
    If the user is an admin, redirects them to the admin panel.
    """
    if 'user_id' not in session or session['user_id'] is None:
        flash('Please login or register to access the recommendation page.', 'info')
        return redirect(url_for('login'))
    
    # Redirect admin users to the admin panel
    if session.get('is_admin'):
        return redirect(url_for('admin_panel'))

    all_service_types = []
    all_locations = []

    if processed_data_global is not None:
        if 'Service Type' in processed_data_global.columns:
             all_service_types = sorted(processed_data_global['Service Type'].unique().tolist())
        else:
            if 'tfidf_service_type' in encoders_global:
                all_service_types = sorted(encoders_global['tfidf_service_type'].get_feature_names_out().tolist())
            else:
                all_service_types = ["Plumber", "Electrician", "Painter", "Carpenter", "Cleaner", "Gardener", "AC Repair", "Appliance Repair", "Cleaning Service", "Pest Control"]

        if 'Location' in processed_data_global.columns:
            all_locations = sorted(processed_data_global['Location'].unique().tolist())

    return render_template('index.html',
                           service_types=all_service_types,
                           locations=all_locations)


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handles the recommendation request using the trained ML model.
    Requires user to be logged in.
    """
    if 'user_id' not in session or session['user_id'] is None:
        if request.is_json:
            return jsonify({'success': False, 'message': 'You must be logged in to get recommendations.'}), 401
        flash('Please login or register to get recommendations.', 'info')
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_service_type = request.form.get('service_type')
        user_location = request.form.get('location')
        user_skills = request.form.get('skills')
        user_rating = request.form.get('rating')
        user_days_available = request.form.get('days_available')
        sort_by = request.form.get('sort_by', 'similarity')
        sort_order = request.form.get('sort_order', 'desc')

        user_input = {
            'service_type': user_service_type,
            'location': user_location,
            'skills': user_skills,
            'rating': user_rating,
            'days_available': user_days_available
        }

        print(f"User request received: {user_input}, Sort By: {sort_by}, Sort Order: {sort_order}")

        recommendations = []
        error_message = None

        if processed_data_global is None or encoders_global is None:
            error_message = "Recommendation system not ready. Please check server logs for ML asset loading errors."
            print(error_message)
        else:
            try:
                recommendations = get_recommendations(user_input, processed_data_global, encoders_global, top_n=10, sort_by=sort_by, sort_order=sort_order)
                if not recommendations:
                    error_message = "No recommendations found for your criteria. Try broadening your search!"
            except Exception as e:
                print(f"Error during recommendation generation: {e}")
                error_message = f"An error occurred while generating recommendations: {e}"

        return render_template('results.html',
                               recommendations=recommendations,
                               user_input=user_input,
                               error_message=error_message,
                               sort_by=sort_by,
                               sort_order=sort_order)

@app.route('/service/<int:service_id>')
def service_detail(service_id):
    """
    Renders the detail page for a specific service provider.
    Fetches service details and reviews from local storage.
    Requires user to be logged in.
    """
    if 'user_id' not in session or session['user_id'] is None:
        flash('Please login or register to view service details.', 'info')
        return redirect(url_for('login'))

    service_details = None
    service_reviews = []
    service_bookings = []

    if processed_data_global is not None:
        service_row = processed_data_global[processed_data_global['ID'] == service_id]
        if not service_row.empty:
            service_details = service_row.iloc[0].to_dict()
            print(f"Found service details for ID {service_id}: {service_details['Name']}")

            # Filter reviews and bookings for this service_id from in-memory lists
            service_reviews = [r for r in reviews_data if r['service_id'] == service_id]
            service_bookings = [b for b in bookings_data if b['service_id'] == service_id]

        else:
            print(f"Service with ID {service_id} not found in processed data.")
    else:
        print("Processed data not loaded. Cannot find service details.")

    return render_template('service_detail.html',
                           service=service_details,
                           reviews=service_reviews,
                           bookings=service_bookings) # Bookings will be empty for regular users on this page

@app.route('/submit_review', methods=['POST'])
def submit_review():
    """
    Handles submission of a new review for a service provider.
    Saves the review to local storage.
    """
    if 'user_id' not in session or session['user_id'] is None:
        return jsonify({'success': False, 'message': 'You must be logged in to submit a review.'}), 401

    try:
        data = request.get_json()
        service_id = int(data.get('service_id'))
        rating = float(data.get('rating'))
        comment = data.get('comment', '')

        if not service_id or not rating:
            return jsonify({'success': False, 'message': 'Service ID and Rating are required.'}), 400

        review_data = {
            'id': f"review_{len(reviews_data) + 1}", # Simple ID
            'service_id': service_id,
            'rating': rating,
            'comment': comment,
            'user_id': session['user_id'],
            'timestamp': datetime.datetime.now().isoformat() # Store as ISO format string
        }

        reviews_data.append(review_data)
        save_json_data(REVIEWS_FILE, reviews_data)
        print(f"Review submitted for service ID {service_id} by user {session['user_id']}.")
        return jsonify({'success': True, 'message': 'Review submitted successfully!'}), 200

    except Exception as e:
        print(f"Error submitting review: {e}")
        return jsonify({'success': False, 'message': f'Error submitting review: {e}'}), 500

@app.route('/submit_booking', methods=['POST'])
def submit_booking():
    """
    Handles submission of a new booking request for a service provider.
    Saves the booking request to local storage.
    """
    if 'user_id' not in session or session['user_id'] is None:
        return jsonify({'success': False, 'message': 'You must be logged in to submit a booking.'}), 401

    try:
        data = request.get_json()
        service_id = int(data.get('service_id'))
        booking_date_str = data.get('booking_date')
        booking_notes = data.get('booking_notes', '')

        if not service_id or not booking_date_str:
            return jsonify({'success': False, 'message': 'Service ID and Preferred Date are required.'}), 400

        booking_data = {
            'id': f"booking_{len(bookings_data) + 1}", # Simple ID
            'service_id': service_id,
            'booking_date': booking_date_str,
            'booking_notes': booking_notes,
            'user_id': session['user_id'],
            'timestamp': datetime.datetime.now().isoformat(), # Store as ISO format string
            'status': 'pending'
        }

        bookings_data.append(booking_data)
        save_json_data(BOOKINGS_FILE, bookings_data)
        print(f"Booking request submitted for service ID {service_id} by user {session['user_id']} for {booking_date_str}.")
        return jsonify({'success': True, 'message': 'Booking request submitted successfully!'}), 200

    except Exception as e:
        print(f"Error submitting booking: {e}")
        return jsonify({'success': False, 'message': f'Error submitting booking: {e}'}), 500

@app.route('/remove_booking', methods=['POST'])
def remove_booking():
    """
    Handles removal of a booking request from local storage.
    Requires admin login.
    """
    global bookings_data
    if not session.get('is_admin'):
        return jsonify({'success': False, 'message': 'Unauthorized: Admin access required.'}), 403

    try:
        data = request.get_json()
        booking_id_to_remove = data.get('booking_id')

        if not booking_id_to_remove:
            return jsonify({'success': False, 'message': 'Booking ID is required.'}), 400

        initial_len = len(bookings_data)
        bookings_data = [b for b in bookings_data if b['id'] != booking_id_to_remove]

        if len(bookings_data) < initial_len:
            save_json_data(BOOKINGS_FILE, bookings_data)
            print(f"Booking {booking_id_to_remove} removed by admin.")
            return jsonify({'success': True, 'message': 'Booking removed successfully.'}), 200
        else:
            return jsonify({'success': False, 'message': 'Booking not found.'}), 404

    except Exception as e:
        print(f"Error removing booking: {e}")
        return jsonify({'success': False, 'message': f'Error removing booking: {e}'}), 500

@app.route('/remove_user', methods=['POST'])
def remove_user():
    """
    Handles removal of a user account and their associated data (reviews, bookings).
    Requires admin login.
    """
    global users_data, reviews_data, bookings_data
    if not session.get('is_admin'):
        return jsonify({'success': False, 'message': 'Unauthorized: Admin access required.'}), 403

    try:
        data = request.get_json()
        user_id_to_remove = data.get('user_id')

        if not user_id_to_remove:
            return jsonify({'success': False, 'message': 'User ID is required.'}), 400

        # Prevent admin from deleting their own account (optional but good practice)
        if user_id_to_remove == session.get('user_id'):
            return jsonify({'success': False, 'message': 'Cannot delete your own admin account.'}), 400

        # Remove user
        initial_users_len = len(users_data)
        users_data = [u for u in users_data if u['id'] != user_id_to_remove]
        if len(users_data) < initial_users_len:
            save_json_data(USERS_FILE, users_data)
            print(f"User {user_id_to_remove} removed by admin.")

            # Remove associated reviews
            initial_reviews_len = len(reviews_data)
            reviews_data = [r for r in reviews_data if r['user_id'] != user_id_to_remove]
            if len(reviews_data) < initial_reviews_len:
                save_json_data(REVIEWS_FILE, reviews_data)
                print(f"Removed {initial_reviews_len - len(reviews_data)} reviews for user {user_id_to_remove}.")

            # Remove associated bookings
            initial_bookings_len = len(bookings_data)
            bookings_data = [b for b in bookings_data if b['user_id'] != user_id_to_remove]
            if len(bookings_data) < initial_bookings_len:
                save_json_data(BOOKINGS_FILE, bookings_data)
                print(f"Removed {initial_bookings_len - len(bookings_data)} bookings for user {user_id_to_remove}.")

            return jsonify({'success': True, 'message': 'User and associated data removed successfully.'}), 200
        else:
            return jsonify({'success': False, 'message': 'User not found.'}), 404

    except Exception as e:
        print(f"Error removing user: {e}")
        return jsonify({'success': False, 'message': f'Error removing user: {e}'}), 500

@app.route('/delete_my_account', methods=['POST'])
def delete_my_account():
    """
    Allows a logged-in user to delete their own account and associated data.
    """
    global users_data, reviews_data, bookings_data
    user_id_to_delete = session.get('user_id')

    if not user_id_to_delete:
        return jsonify({'success': False, 'message': 'You must be logged in to delete your account.'}), 401

    # Prevent admin from deleting their account via this route (they have a separate admin removal)
    if user_id_to_delete == ADMIN_USERNAME:
        return jsonify({'success': False, 'message': 'Admin accounts cannot be deleted via this route.'}), 403

    try:
        # Remove user
        initial_users_len = len(users_data)
        users_data = [u for u in users_data if u['id'] != user_id_to_delete]
        if len(users_data) < initial_users_len:
            save_json_data(USERS_FILE, users_data)
            print(f"User {user_id_to_delete} deleted their own account.")

            # Remove associated reviews
            initial_reviews_len = len(reviews_data)
            reviews_data = [r for r in reviews_data if r['user_id'] == user_id_to_delete]
            if len(reviews_data) < initial_reviews_len:
                save_json_data(REVIEWS_FILE, reviews_data)
                print(f"Removed {initial_reviews_len - len(reviews_data)} reviews for deleted user {user_id_to_delete}.")

            # Remove associated bookings
            initial_bookings_len = len(bookings_data)
            bookings_data = [b for b in bookings_data if b['user_id'] == user_id_to_delete]
            if len(bookings_data) < initial_bookings_len:
                save_json_data(BOOKINGS_FILE, bookings_data)
                print(f"Removed {initial_bookings_len - len(bookings_data)} bookings for deleted user {user_id_to_delete}.")

            # Clear session and log out the user
            session.pop('user_id', None)
            session.pop('is_admin', None)
            return jsonify({'success': True, 'message': 'Your account and all associated data have been deleted.'}), 200
        else:
            return jsonify({'success': False, 'message': 'Account not found.'}), 404

    except Exception as e:
        print(f"Error deleting user account: {e}")
        return jsonify({'success': False, 'message': f'Error deleting account: {e}'}), 500


@app.route('/admin')
def admin_panel():
    """
    Renders a basic admin panel to view all service providers, reviews, and bookings.
    Requires admin login.
    """
    if not session.get('is_admin'):
        flash('You must be logged in as an administrator to access this page.', 'danger')
        return redirect(url_for('login'))

    all_services = []
    if processed_data_global is not None:
        all_services = processed_data_global[['ID', 'Name', 'Service Type', 'Location', 'Rating']].to_dict(orient='records')

    # Use the in-memory lists
    all_users = users_data # Get all users for admin panel
    all_reviews = reviews_data
    all_bookings = bookings_data

    return render_template('admin.html',
                           services=all_services,
                           users=all_users, # Pass users to the template
                           reviews=all_reviews,
                           bookings=all_bookings)


if __name__ == '__main__':
    # Load all local data (users, reviews, bookings) on app startup
    load_all_local_data()
    load_ml_assets()
    # Open the web browser automatically
    # Only open browser if not in a reloader process (prevents opening two tabs)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open_new("http://127.0.0.1:5000/welcome")
    app.run(debug=True)
