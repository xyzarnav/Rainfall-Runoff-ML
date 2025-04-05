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
});