// This file contains client-side JavaScript functionality for the Home Service Recommender.

document.addEventListener('DOMContentLoaded', function() {
    // --- Copy Contact Number Functionality ---
    const copyButtons = document.querySelectorAll('.copy-contact-btn');

    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const contactNumber = this.getAttribute('data-contact');
            const targetId = this.getAttribute('data-target'); // This will be 'contact-X' or 'contact-detail'
            const copiedMessageSpan = document.getElementById(`copied-message-${targetId.split('-')[1] || 'detail'}`);

            // Create a temporary textarea element to copy text
            const tempTextArea = document.createElement('textarea');
            tempTextArea.value = contactNumber;
            document.body.appendChild(tempTextArea);
            tempTextArea.select();

            try {
                // Execute the copy command
                document.execCommand('copy');
                console.log('Contact number copied:', contactNumber);

                // Show "Copied!" message
                if (copiedMessageSpan) {
                    copiedMessageSpan.classList.remove('d-none'); // Show the message
                    setTimeout(() => {
                        copiedMessageSpan.classList.add('d-none'); // Hide after 2 seconds
                    }, 2000);
                }
            } catch (err) {
                console.error('Failed to copy contact number:', err);
            } finally {
                document.body.removeChild(tempTextArea); // Clean up the temporary element
            }
        });
    });

    // --- Review Form Submission (AJAX) ---
    const reviewForm = document.getElementById('reviewForm');
    if (reviewForm) {
        reviewForm.addEventListener('submit', async function(event) {
            event.preventDefault(); // Prevent default form submission

            const serviceId = document.getElementById('serviceIdReview').value;
            const rating = document.getElementById('reviewRating').value;
            const comment = document.getElementById('reviewComment').value;
            const reviewMessageDiv = document.getElementById('reviewMessage');

            reviewMessageDiv.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div> Submitting review...';
            reviewMessageDiv.className = 'mt-3 text-center'; // Reset classes

            try {
                const response = await fetch('/submit_review', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        service_id: parseInt(serviceId),
                        rating: parseFloat(rating),
                        comment: comment
                    })
                });

                const result = await response.json();

                if (result.success) {
                    reviewMessageDiv.innerHTML = `<div class="alert alert-success rounded-pill" role="alert">${result.message}</div>`;
                    reviewMessageDiv.className = 'mt-3';
                    document.getElementById('reviewRating').value = '';
                    document.getElementById('reviewComment').value = '';
                    setTimeout(() => {
                        window.location.reload(); // Reload to show new review
                    }, 1500);
                } else {
                    reviewMessageDiv.innerHTML = `<div class="alert alert-danger rounded-pill" role="alert">${result.message}</div>`;
                    reviewMessageDiv.className = 'mt-3';
                }
            } catch (error) {
                console.error('Error submitting review:', error);
                reviewMessageDiv.innerHTML = `<div class="alert alert-danger rounded-pill" role="alert">An unexpected error occurred.</div>`;
                reviewMessageDiv.className = 'mt-3';
            }
        });
    }

    // --- Booking Form Submission (AJAX) ---
    const bookingForm = document.getElementById('bookingForm');
    if (bookingForm) {
        bookingForm.addEventListener('submit', async function(event) {
            event.preventDefault(); // Prevent default form submission

            const serviceId = document.getElementById('serviceIdBooking').value;
            const bookingDate = document.getElementById('bookingDate').value;
            const bookingNotes = document.getElementById('bookingNotes').value;
            const bookingMessageDiv = document.getElementById('bookingMessage');

            if (!bookingDate) {
                bookingMessageDiv.innerHTML = '<div class="alert alert-warning rounded-pill" role="alert">Please select a preferred date.</div>';
                bookingMessageDiv.className = 'mt-3';
                return;
            }

            bookingMessageDiv.innerHTML = '<div class="spinner-border text-success" role="status"><span class="visually-hidden">Loading...</span></div> Submitting booking request...';
            bookingMessageDiv.className = 'mt-3 text-center'; // Reset classes

            try {
                const response = await fetch('/submit_booking', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        service_id: parseInt(serviceId),
                        booking_date: bookingDate,
                        booking_notes: bookingNotes
                    })
                });

                const result = await response.json();

                if (result.success) {
                    bookingMessageDiv.innerHTML = `<div class="alert alert-success rounded-pill" role="alert">${result.message}</div>`;
                    bookingMessageDiv.className = 'mt-3';
                    document.getElementById('bookingDate').value = '';
                    document.getElementById('bookingNotes').value = '';
                } else {
                    bookingMessageDiv.innerHTML = `<div class="alert alert-danger rounded-pill" role="alert">${result.message}</div>`;
                    bookingMessageDiv.className = 'mt-3';
                }
            } catch (error) {
                console.error('Error submitting booking:', error);
                bookingMessageDiv.innerHTML = `<div class="alert alert-danger rounded-pill" role="alert">An unexpected error occurred.</div>`;
                bookingMessageDiv.className = 'mt-3';
            }
        });
    }

    // --- Remove Booking Functionality (AJAX) ---
    const removeBookingButtons = document.querySelectorAll('.remove-booking-btn');
    removeBookingButtons.forEach(button => {
        button.addEventListener('click', async function() {
            const bookingId = this.getAttribute('data-booking-id');
            if (confirm(`Are you sure you want to remove booking ID: ${bookingId}?`)) { // Using confirm for simplicity
                try {
                    const response = await fetch('/remove_booking', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ booking_id: bookingId })
                    });

                    const result = await response.json();

                    if (result.success) {
                        alert(result.message); // Using alert for simplicity
                        // Remove the row from the table
                        const rowToRemove = document.getElementById(`booking-row-${bookingId}`);
                        if (rowToRemove) {
                            rowToRemove.remove();
                        }
                    } else {
                        alert(`Error: ${result.message}`); // Using alert for simplicity
                    }
                } catch (error) {
                    console.error('Error removing booking:', error);
                    alert('An error occurred while trying to remove the booking.'); // Using alert for simplicity
                }
            }
        });
    });

    // --- Remove User Functionality (Admin Panel AJAX) ---
    const removeUserButtons = document.querySelectorAll('.remove-user-btn');
    removeUserButtons.forEach(button => {
        button.addEventListener('click', async function() {
            const userId = this.getAttribute('data-user-id');
            if (confirm(`Are you sure you want to remove user ID: ${userId} and all their associated data (reviews, bookings)? This action cannot be undone.`)) {
                try {
                    const response = await fetch('/remove_user', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ user_id: userId })
                    });

                    const result = await response.json();

                    if (result.success) {
                        alert(result.message);
                        // Remove the row from the table
                        const rowToRemove = document.getElementById(`user-row-${userId}`);
                        if (rowToRemove) {
                            rowToRemove.remove();
                        }
                    } else {
                        alert(`Error: ${result.message}`);
                    }
                } catch (error) {
                    console.error('Error removing user:', error);
                    alert('An error occurred while trying to remove the user.');
                }
            }
        });
    });

    // --- Delete My Account Functionality (User AJAX) ---
    const deleteMyAccountBtn = document.getElementById('deleteMyAccountBtn');
    if (deleteMyAccountBtn) {
        deleteMyAccountBtn.addEventListener('click', async function() {
            if (confirm('Are you sure you want to delete your account? This will remove all your reviews and bookings and cannot be undone.')) {
                try {
                    const response = await fetch('/delete_my_account', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({}) // No specific user_id needed, server uses session
                    });

                    const result = await response.json();

                    if (result.success) {
                        alert(result.message);
                        window.location.href = '/welcome'; // Redirect to welcome page after deletion
                    } else {
                        alert(`Error: ${result.message}`);
                    }
                } catch (error) {
                    console.error('Error deleting my account:', error);
                    alert('An error occurred while trying to delete your account.');
                }
            }
        });
    }
});
