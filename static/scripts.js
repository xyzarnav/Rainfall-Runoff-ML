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

    // Form submission handler
    form.addEventListener('submit', function (e) {
        e.preventDefault();

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

                console.log("Server response:", data); // Debug to see what's coming back

                // Format model name properly
                let modelName = "Unknown Model";
                if (data.model_type === 'lstm') {
                    modelName = 'LSTM-Attention';
                } else if (data.model_type === 'gru') {
                    modelName = 'GRU-Attention';
                } else if (data.model_type === 'transformer') {
                    modelName = 'Transformer';
                } else if (data.model_type === 'pca-lstm') {
                    modelName = 'PCA-Based LSTM-Attention';
                }

                // Add window size to model name if available
                if (data.window_size) {
                    modelName += ` (Window Size: ${data.window_size})`;
                }

                // Update metrics with correct property names
                document.getElementById('model-name').textContent = modelName;
                document.getElementById('predicted-runoff').textContent =
                    (data.total_predicted_runoff).toFixed(2) + ' m³/s';
                document.getElementById('actual-runoff').textContent =
                    (data.total_actual_runoff).toFixed(2) + ' m³/s';
                document.getElementById('mse').textContent =
                    (data.mse).toFixed(6);
                document.getElementById('mae').textContent =
                    (data.mae).toFixed(6);
                document.getElementById('r2').textContent =
                    (data.r2_score).toFixed(6);

                // Update plot images with correct URLs
                document.getElementById('line-plot').src = data.line_plot_url;
                document.getElementById('scatter-plot').src = data.scatter_plot_url;
                document.getElementById('bar-plot').src = data.bar_plot_url;
                document.getElementById('residuals-plot').src = data.residuals_plot_url;

                // Add error handling for images
                const imageElements = [
                    document.getElementById('line-plot'),
                    document.getElementById('scatter-plot'),
                    document.getElementById('bar-plot'),
                    document.getElementById('residuals-plot')
                ];

                imageElements.forEach(img => {
                    img.onerror = function () {
                        this.onerror = null; // Prevent infinite loop
                        const errorDiv = document.createElement('div');
                        errorDiv.className = 'alert alert-warning mt-3';
                        errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>Image could not be loaded.`;
                        this.parentNode.appendChild(errorDiv);
                        this.style.display = 'none';
                    };
                });

                // Show results with animation
                metrics.style.display = 'block';
                plots.style.display = 'block';
                printSection.style.display = 'block';
            })
            .catch(error => {
                loading.style.display = 'none';
                console.error('Error:', error);
                alert('Error processing data: ' + error.message);
            });
    });

    // Print report functionality
    document.getElementById('print-report').addEventListener('click', function () {
        window.print();
    });

    // Download PDF functionality
    document.getElementById('download-pdf').addEventListener('click', function () {
        alert('PDF download functionality will be implemented here');
        // Implementation for PDF download would go here
    });
});



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

    // Form submission handler
    form.addEventListener('submit', function (e) {
        e.preventDefault();

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
                console.log("Server response:", data); // Debug logging

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
                }

                // Add window size if available
                if (data.window_size) {
                    modelName += ` (Window Size: ${data.window_size})`;
                }

                // Update metrics
                document.getElementById('model-name').textContent = modelName;
                document.getElementById('predicted-runoff').textContent =
                    (data.total_predicted_runoff).toFixed(2) + ' m³/s';
                document.getElementById('actual-runoff').textContent =
                    (data.total_actual_runoff).toFixed(2) + ' m³/s';
                document.getElementById('mse').textContent =
                    (data.mse).toFixed(6);
                document.getElementById('mae').textContent =
                    (data.mae).toFixed(6);
                document.getElementById('r2').textContent =
                    (data.r2_score).toFixed(6);

                // Update plot images with correct URL properties
                document.getElementById('line-plot').src = data.line_plot_url;
                document.getElementById('scatter-plot').src = data.scatter_plot_url;
                document.getElementById('bar-plot').src = data.bar_plot_url;
                document.getElementById('residuals-plot').src = data.residuals_plot_url;

                // Add error handling for images
                const imageElements = document.querySelectorAll('.tab-pane img');
                imageElements.forEach(img => {
                    img.onerror = function () {
                        console.error(`Failed to load image: ${this.src}`);
                        this.style.display = 'none';

                        // Create error message if not already present
                        if (!this.nextElementSibling || !this.nextElementSibling.classList.contains('alert')) {
                            const errorDiv = document.createElement('div');
                            errorDiv.className = 'alert alert-warning mt-3';
                            errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>Image could not be loaded.`;
                            this.parentNode.appendChild(errorDiv);
                        }
                    };
                });

                // Show results with animation
                metrics.style.display = 'block';
                plots.style.display = 'block';
                printSection.style.display = 'block';
            })
            .catch(error => {
                loading.style.display = 'none';
                alert('Error processing data: ' + error.message);
                console.error('Error:', error);
            });
    });

    // Print report functionality
    document.getElementById('print-report').addEventListener('click', function () {
        window.print();
    });

    // Download PDF functionality
    document.getElementById('download-pdf').addEventListener('click', function () {
        alert('PDF download functionality will be implemented here');
        // Implementation for PDF download would go here
    });
});
document.addEventListener('DOMContentLoaded', function() {
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
    
    // Form submission handler
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
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
            console.log("Server response:", data); // Debug logging
            
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