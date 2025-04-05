document.addEventListener('DOMContentLoaded', function () {
    // Initialize AOS animation library
    AOS.init({
        duration: 800,
        easing: 'ease-in-out',
        once: true
    });
    
    const form = document.getElementById('upload-form');
    const loading = document.getElementById('loading');
    const metrics = document.getElementById('metrics');
    const plots = document.getElementById('plots');
    const printSection = document.getElementById('print-section');
    
    // Form submission handler - prevent double submissions
    let isSubmitting = false;
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Prevent multiple submissions
        if (isSubmitting) {
            console.log('Form already being submitted, ignoring additional click');
            return;
        }
        
        isSubmitting = true;
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        
        // Show loading indicator
        loading.style.display = 'block';
        metrics.style.display = 'none';
        plots.style.display = 'none';
        printSection.style.display = 'none';
        
        // Get form data
        const formData = new FormData(form);
        
        // Make AJAX request
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            loading.style.display = 'none';
            
            try {
                // Format model name nicely
                let modelName = "Unknown Model";
                if (data.model_type === 'lstm') {
                    modelName = 'LSTM-Attention';
                } else if (data.model_type === 'gru') {
                    modelName = 'GRU-Attention';
                } else if (data.model_type === 'transformer') {
                    modelName = 'Transformer';
                } else if (data.model_type === 'pca-lstm') {
                    modelName = 'PCA-Based LSTM-Attention';
                } else if (data.model_type === 'enhanced_gru') {
                    modelName = 'Enhanced GRU-Attention';
                }
                
                // Add window size if available
                if (data.window_size) {
                    modelName += ` (Window Size: ${data.window_size})`;
                }
                
                // Update metrics with robust error handling
                document.getElementById('model-name').textContent = modelName;
                
                // Safety check for numeric values
                const safeNumber = (value, decimals = 2) => {
                    return (typeof value === 'number' && !isNaN(value)) ? 
                        value.toFixed(decimals) : 'N/A';
                };
                
                document.getElementById('predicted-runoff').textContent = 
                    safeNumber(data.total_predicted_runoff, 2) + ' m³/s';
                document.getElementById('actual-runoff').textContent = 
                    safeNumber(data.total_actual_runoff, 2) + ' m³/s';
                document.getElementById('mse').textContent = 
                    safeNumber(data.mse, 6);
                document.getElementById('mae').textContent = 
                    safeNumber(data.mae, 6);
                document.getElementById('r2').textContent = 
                    safeNumber(data.r2_score, 6);
                
                // Update plot images with proper error handling
                function setImageWithFallback(imgElement, url) {
                    if (!url) {
                        showImageError(imgElement);
                        return;
                    }
                    
                    imgElement.src = url;
                    imgElement.onerror = () => showImageError(imgElement);
                }
                
                function showImageError(imgElement) {
                    console.error(`Failed to load image: ${imgElement.src}`);
                    imgElement.style.display = 'none';
                    
                    const container = imgElement.parentElement;
                    if (!container.querySelector('.alert')) {
                        const errorDiv = document.createElement('div');
                        errorDiv.className = 'alert alert-warning mt-3';
                        errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>
                            Image could not be loaded.`;
                        container.appendChild(errorDiv);
                    }
                }
                
                // Set images with proper error handling
                setImageWithFallback(document.getElementById('line-plot'), data.line_plot_url);
                setImageWithFallback(document.getElementById('scatter-plot'), data.scatter_plot_url);
                setImageWithFallback(document.getElementById('bar-plot'), data.bar_plot_url);
                setImageWithFallback(document.getElementById('residuals-plot'), data.residuals_plot_url);
                
                // Show results sections
                metrics.style.display = 'block';
                plots.style.display = 'block';
                printSection.style.display = 'block';
            
            } catch (error) {
                console.error("Error processing response:", error);
                alert('Error processing data: ' + error.message);
            }
        })
        .catch(error => {
            loading.style.display = 'none';
            console.error('Error:', error);
            alert('Error contacting the server: ' + error.message);
        })
        .finally(() => {
            // Re-enable form submission after processing completes
            isSubmitting = false;
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fas fa-cogs me-2"></i>Analyze Data';
        });
    });
    
    // Print report functionality
    document.getElementById('print-report').addEventListener('click', function() {
        window.print();
    });
    
    // Download PDF functionality
    document.getElementById('download-pdf').addEventListener('click', function() {
        alert('PDF download functionality will be implemented here');
        // PDF implementation would go here
    });
        
    // Add event listener for the clear data button
    const clearDataBtn = document.getElementById('clearData');
    if (clearDataBtn) {
        clearDataBtn.addEventListener('click', clearStoredData);
    }

    // Add event listeners for both clear data buttons
    const clearDataButtons = ['navClearData', 'heroClearData'];
    clearDataButtons.forEach(buttonId => {
        const button = document.getElementById(buttonId);
        if (button) {
            button.addEventListener('click', clearStoredData);
        }
    });

    // Simplified clear data button initialization
    function initializeClearButtons() {
        // Get all clear data buttons
        const buttons = document.querySelectorAll('button[id$="ClearData"]');
        
        buttons.forEach(button => {
            button.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                clearStoredData();
            });
        });

        console.log('Clear data buttons initialized:', Array.from(buttons).map(btn => btn.id));
    }

    // Initialize clear buttons
    initializeClearButtons();
});

function clearStoredData() {
    if (confirm('Are you sure you want to clear all data? This action cannot be undone.')) {
        try {
            // Clear all stored data
            localStorage.clear();
            sessionStorage.clear();
            
            // Reset form
            const uploadForm = document.getElementById('upload-form');
            if (uploadForm) uploadForm.reset();
            
            // Reset all display elements
            ['metrics', 'plots', 'print-section'].forEach(id => {
                const element = document.getElementById(id);
                if (element) element.style.display = 'none';
            });
            
            // Clear metric values
            ['model-name', 'predicted-runoff', 'actual-runoff', 'mse', 'mae', 'r2'].forEach(id => {
                const element = document.getElementById(id);
                if (element) element.textContent = '';
            });
            
            // Clear plot images
            ['line-plot', 'scatter-plot', 'bar-plot', 'residuals-plot'].forEach(id => {
                const element = document.getElementById(id);
                if (element) element.src = '';
            });
            
            // Hide loading spinner
            const loading = document.getElementById('loading');
            if (loading) loading.style.display = 'none';
            
            // Clear console output
            console.clear();
            
            // Clear any existing model state
            if (window.modelState) {
                window.modelState = null;
            }
            
            // Make a request to clear server-side data
            fetch('/clear_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showToast('All data has been cleared successfully!', 'success');
                } else {
                    throw new Error('Failed to clear server data');
                }
            })
            .catch(error => {
                console.error('Error clearing server data:', error);
                showToast('Warning: Client data cleared but server data may remain', 'warning');
            });

        } catch (error) {
            console.error('Error clearing data:', error);
            showToast('Error clearing data. Please try again.', 'error');
        }
    }
}

// Add toast notification function
function showToast(message, type = 'success') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        `;
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.style.cssText = `
        background: ${type === 'success' ? '#4CAF50' : '#f44336'};
        color: white;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 10px;
        min-width: 250px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    `;
    
    toast.textContent = message;
    
    // Add to container
    toastContainer.appendChild(toast);
    
    // Remove after 3 seconds
    setTimeout(() => {
        toast.remove();
    }, 3000);
}