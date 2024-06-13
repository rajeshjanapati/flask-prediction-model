import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process_data', methods=['POST'])

def process_data():
    data = request.json
    number_value = int(data['number'])  # Extract the number value received from the client

    # Assuming you have the following data
    product_ids = [110002201, 110002202, 110002203, 110002204, 110002205, 110002206, 110002207, 110002208, 110002209, 110002210, 110002211, 110002212, 110002213, 110002214, 110002215, 110002216, 110002217, 110002218, 110002219, 110002220, 110002221, 110002222, 110002223, 110002224, 110002225, 110002226, 110002227, 110002228, 110002229, 110002230, 110002231, 110002232, 110002233, 110002234, 110002235, 110002236, 110002237, 110002238, 110002239, 110002240, 110002241, 110002242, 110002243, 110002244, 110002245, 110002246, 110002247, 110002248, 110002249, 110002250, 110002251, 110002252, 110002253, 110002254, 110002255, 110002256, 110002257, 110002258, 110002259, 110002260, 110002261, 110002262, 110002263, 110002264, 110002265, 110002266, 110002267, 110002268, 110002269, 110002270, 110002271, 110002272, 110002273, 110002274, 110002275, 110002276, 110002277, 110002278, 110002279, 110002280, 110002281, 110002282, 110002283, 110002284, 110002285, 110002286, 110002287, 110002288, 110002289, 110002290, 110002291, 110002292, 110002293, 110002294, 110002295, 110002296, 110002297, 110002298, 110002299, 110002300, 110002301, 110002302, 110002303, 110002304, 110002305, 110002306, 110002307, 110002308, 110002309, 110002310]  # List of product IDs
    product_statuses = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # List of product statuses
    costs = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]  # List of costs



    # Convert lists to numpy arrays
    X = np.array([product_ids, product_statuses]).T
    y = np.array(costs)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Specify the number of products for prediction
    num_products_24hrs_200 = 200

    # Generate product IDs and statuses for the specified number of products (200 products)
    predicted_product_ids_200 = [110002201, 110002202, 110002203, 110002204, 110002205, 110002206, 110002207, 110002208, 110002209, 110002210, 110002211, 110002212, 110002213, 110002214, 110002215, 110002216, 110002217, 110002218, 110002219, 110002220, 110002221, 110002222, 110002223, 110002224, 110002225, 110002226, 110002227, 110002228, 110002229, 110002230, 110002231, 110002232, 110002233, 110002234, 110002235, 110002236, 110002237, 110002238, 110002239, 110002240, 110002241, 110002242, 110002243, 110002244, 110002245, 110002246, 110002247, 110002248, 110002249, 110002250, 110002251, 110002252, 110002253, 110002254, 110002255, 110002256, 110002257, 110002258, 110002259, 110002260, 110002261, 110002262, 110002263, 110002264, 110002265, 110002266, 110002267, 110002268, 110002269, 110002270, 110002271, 110002272, 110002273, 110002274, 110002275, 110002276, 110002277, 110002278, 110002279, 110002280, 110002281, 110002282, 110002283, 110002284, 110002285, 110002286, 110002287, 110002288, 110002289, 110002290, 110002291, 110002292, 110002293, 110002294, 110002295, 110002296, 110002297, 110002298, 110002299, 110002300, 110002301, 110002302, 110002303, 110002304, 110002305, 110002306, 110002307, 110002308, 110002309, 110002310]  # List of product IDs
    predicted_product_statuses_200 = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # List of product statuses

    # Predict the cost for the specified number of products (200 products)
    predicted_costs_200 = model.predict(np.array([predicted_product_ids_200, predicted_product_statuses_200]).T)

    # Calculate success and failure percentages for 200 products
    predicted_statuses_200 = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]  # List of product statuses
    # actual_statuses = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1]  # List of product statuses
    actual_statuses_200 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # List of product statuses

    successful_predictions_200 = sum(1 for pred, actual in zip(predicted_statuses_200, actual_statuses_200) if pred == actual)
    failed_predictions_200 = len(predicted_statuses_200) - successful_predictions_200
    total_predictions_200 = len(predicted_statuses_200)
    success_percentage_200 = (successful_predictions_200 / total_predictions_200) * 100
    failure_percentage_200 = (failed_predictions_200 / total_predictions_200) * 100

    # Output the results for 200 products
    # print(f'Predicted Cost for {num_products_24hrs_200} Products: {np.mean(predicted_costs_200) * num_products_24hrs_200}')
    # print(f'Success Percentage for {num_products_24hrs_200} Products: {success_percentage_200}%')
    # print(f'Failure Percentage for {num_products_24hrs_200} Products: {failure_percentage_200}%')

    # Now, let's predict for 400 products in the same 24-hour period
    # We will scale the success and failure percentages obtained for 200 products
    num_products_24hrs_400 = number_value

    # Scale the success and failure percentages for 400 products based on the 200 products' data
    scaled_success_percentage_400 = min(success_percentage_200 + np.random.uniform(-2, 2), 100)  # Adding a random variation within a reasonable range
    scaled_failure_percentage_400 = min(failure_percentage_200 + np.random.uniform(-2, 2), 100)  # Adding a random variation within a reasonable range

    # Output the scaled success and failure percentages for 400 products
    print(f'Success Percentage for {num_products_24hrs_400} Products: {scaled_success_percentage_400}%')
    print(f'Failure Percentage for {num_products_24hrs_400} Products: {scaled_failure_percentage_400}%')

    # Predict the cost for the additional 200 products
#     predicted_costs_400 = predicted_costs_200 * 2  # Assuming the same cost for each product
    predicted_costs_400 = predicted_costs_200
    
    # Output the predicted cost for 400 products
    print(f'Predicted Cost for {num_products_24hrs_400} Products: {np.mean(predicted_costs_400) * num_products_24hrs_400}')

    # Calculate accuracy for the first 200 products
    accuracy_200 = (sum(1 for pred, actual in zip(predicted_statuses_200, actual_statuses_200) if pred == actual) / len(actual_statuses_200)) * 100

    # Estimate accuracy for the remaining 200 products
    # For simplicity, we assume the same accuracy rate continues
    accuracy_remaining_200 = accuracy_200  # Adjust this based on the characteristics of your data

    # Output the result
    print(f'Accuracy for {num_products_24hrs_400} Products: {accuracy_remaining_200}%')
    
    scaled_failure_percentage_400 = 100 - scaled_success_percentage_400
    Predicted_Cost = int(np.mean(predicted_costs_400) * num_products_24hrs_400)
    

    response_data = {
#         'Success Percentage': f"{scaled_success_percentage_400:.2f}%",
        'Success Percentage': str(f"{scaled_success_percentage_400:.2f}%"),
        'Failure Percentage' : str(f"{scaled_failure_percentage_400:.2f}%"),
        'Predicted_Cost': f"${Predicted_Cost}.00",
        'Accuracy': str(f"{accuracy_remaining_200:.2f}%")
        }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=8000)  # Run the Flask app




























# import numpy as np
# from sklearn.linear_model import LinearRegression
# from flask import Flask, request, jsonify
# 
# app = Flask(__name__)
# 
# @app.route('/process_data', methods=['POST'])
# 
# def process_data():
#     data = request.json
#     number_value = int(data['number'])  # Extract the number value received from the client
# 
#     # Assuming you have the following data
#     product_ids = [110002201, 110002202, 110002203, 110002204, 110002205, 110002206, 110002207, 110002208, 110002209, 110002210, 110002211, 110002212, 110002213, 110002214, 110002215, 110002216, 110002217, 110002218, 110002219, 110002220, 110002221, 110002222, 110002223, 110002224, 110002225, 110002226, 110002227, 110002228, 110002229, 110002230, 110002231, 110002232, 110002233, 110002234, 110002235, 110002236, 110002237, 110002238, 110002239, 110002240, 110002241, 110002242, 110002243, 110002244, 110002245, 110002246, 110002247, 110002248, 110002249, 110002250, 110002251, 110002252, 110002253, 110002254, 110002255, 110002256, 110002257, 110002258, 110002259, 110002260, 110002261, 110002262, 110002263, 110002264, 110002265, 110002266, 110002267, 110002268, 110002269, 110002270, 110002271, 110002272, 110002273, 110002274, 110002275, 110002276, 110002277, 110002278, 110002279, 110002280, 110002281, 110002282, 110002283, 110002284, 110002285, 110002286, 110002287, 110002288, 110002289, 110002290, 110002291, 110002292, 110002293, 110002294, 110002295, 110002296, 110002297, 110002298, 110002299, 110002300, 110002301, 110002302, 110002303, 110002304, 110002305, 110002306, 110002307, 110002308, 110002309, 110002310]  # List of product IDs
#     product_statuses = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # List of product statuses
#     costs = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]  # List of costs
# 
# 
# 
#     # Convert lists to numpy arrays
#     X = np.array([product_ids, product_statuses]).T
#     y = np.array(costs)
# 
#     # Train the linear regression model
#     model = LinearRegression()
#     model.fit(X, y)
# 
#     # Specify the number of products for prediction
#     num_products_24hrs_200 = 200
# 
#     # Generate product IDs and statuses for the specified number of products (200 products)
#     predicted_product_ids_200 = [110002201, 110002202, 110002203, 110002204, 110002205, 110002206, 110002207, 110002208, 110002209, 110002210, 110002211, 110002212, 110002213, 110002214, 110002215, 110002216, 110002217, 110002218, 110002219, 110002220, 110002221, 110002222, 110002223, 110002224, 110002225, 110002226, 110002227, 110002228, 110002229, 110002230, 110002231, 110002232, 110002233, 110002234, 110002235, 110002236, 110002237, 110002238, 110002239, 110002240, 110002241, 110002242, 110002243, 110002244, 110002245, 110002246, 110002247, 110002248, 110002249, 110002250, 110002251, 110002252, 110002253, 110002254, 110002255, 110002256, 110002257, 110002258, 110002259, 110002260, 110002261, 110002262, 110002263, 110002264, 110002265, 110002266, 110002267, 110002268, 110002269, 110002270, 110002271, 110002272, 110002273, 110002274, 110002275, 110002276, 110002277, 110002278, 110002279, 110002280, 110002281, 110002282, 110002283, 110002284, 110002285, 110002286, 110002287, 110002288, 110002289, 110002290, 110002291, 110002292, 110002293, 110002294, 110002295, 110002296, 110002297, 110002298, 110002299, 110002300, 110002301, 110002302, 110002303, 110002304, 110002305, 110002306, 110002307, 110002308, 110002309, 110002310]  # List of product IDs
#     predicted_product_statuses_200 = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # List of product statuses
# 
#     # Predict the cost for the specified number of products (200 products)
#     predicted_costs_200 = model.predict(np.array([predicted_product_ids_200, predicted_product_statuses_200]).T)
# 
#     # Calculate success and failure percentages for 200 products
#     predicted_statuses_200 = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # List of product statuses
#     # actual_statuses = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1]  # List of product statuses
#     actual_statuses_200 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # List of product statuses
# 
#     successful_predictions_200 = sum(1 for pred, actual in zip(predicted_statuses_200, actual_statuses_200) if pred == actual)
#     failed_predictions_200 = len(predicted_statuses_200) - successful_predictions_200
#     total_predictions_200 = len(predicted_statuses_200)
#     success_percentage_200 = (successful_predictions_200 / total_predictions_200) * 100
#     failure_percentage_200 = (failed_predictions_200 / total_predictions_200) * 100
# 
#     # Output the results for 200 products
#     # print(f'Predicted Cost for {num_products_24hrs_200} Products: {np.mean(predicted_costs_200) * num_products_24hrs_200}')
#     # print(f'Success Percentage for {num_products_24hrs_200} Products: {success_percentage_200}%')
#     # print(f'Failure Percentage for {num_products_24hrs_200} Products: {failure_percentage_200}%')
# 
#     # Now, let's predict for 400 products in the same 24-hour period
#     # We will scale the success and failure percentages obtained for 200 products
#     num_products_24hrs_400 = number_value
# 
#     # Scale the success and failure percentages for 400 products based on the 200 products' data
#     scaled_success_percentage_400 = min(success_percentage_200 + np.random.uniform(-2, 2), 100)  # Adding a random variation within a reasonable range
#     scaled_failure_percentage_400 = min(failure_percentage_200 + np.random.uniform(-2, 2), 100)  # Adding a random variation within a reasonable range
# 
#     # Output the scaled success and failure percentages for 400 products
#     print(f'Success Percentage for {num_products_24hrs_400} Products: {scaled_success_percentage_400}%')
#     print(f'Failure Percentage for {num_products_24hrs_400} Products: {scaled_failure_percentage_400}%')
# 
#     # Predict the cost for the additional 200 products
#     predicted_costs_400 = predicted_costs_200 * 2  # Assuming the same cost for each product
# 
#     # Output the predicted cost for 400 products
#     print(f'Predicted Cost for {num_products_24hrs_400} Products: {np.mean(predicted_costs_400) * num_products_24hrs_400}')
# 
#     # Calculate accuracy for the first 200 products
#     accuracy_200 = (sum(1 for pred, actual in zip(predicted_statuses_200, actual_statuses_200) if pred == actual) / len(actual_statuses_200)) * 100
# 
#     # Estimate accuracy for the remaining 200 products
#     # For simplicity, we assume the same accuracy rate continues
#     accuracy_remaining_200 = accuracy_200  # Adjust this based on the characteristics of your data
# 
#     # Output the result
#     print(f'Accuracy for {num_products_24hrs_400} Products: {accuracy_remaining_200}%')
#     
# 
#     response_data = {
#         'Success Percentage': str(scaled_success_percentage_400) + "%",
#         'Failure Percentage' : str(scaled_failure_percentage_400) + "%",
#         'Predicted_Cost': np.mean(predicted_costs_400) * num_products_24hrs_400,
#         'Accuracy': str(accuracy_remaining_200) + "%"
#         }
#     return jsonify(response_data)
# 
# 
# if __name__ == '__main__':
#     app.run(debug=True)  # Run the Flask app
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # import numpy as np
# # from sklearn.linear_model import LinearRegression
# 
# # # Assuming you have the following data
# # # Features: product_id, product_status
# # # Target: cost
# # # You should replace the following arrays with your actual data
# # product_ids = [110002201, 110002202, 110002203, 110002204, 110002205, 110002206, 110002207, 110002208, 110002209, 110002210, 110002211, 110002212, 110002213, 110002214, 110002215, 110002216, 110002217, 110002218, 110002219, 110002220, 110002221, 110002222, 110002223, 110002224, 110002225, 110002226, 110002227, 110002228, 110002229, 110002230, 110002231, 110002232, 110002233, 110002234, 110002235, 110002236, 110002237, 110002238, 110002239, 110002240, 110002241, 110002242, 110002243, 110002244, 110002245, 110002246, 110002247, 110002248, 110002249, 110002250, 110002251, 110002252, 110002253, 110002254, 110002255, 110002256, 110002257, 110002258, 110002259, 110002260, 110002261, 110002262, 110002263, 110002264, 110002265, 110002266, 110002267, 110002268, 110002269, 110002270, 110002271, 110002272, 110002273, 110002274, 110002275, 110002276, 110002277, 110002278, 110002279, 110002280, 110002281, 110002282, 110002283, 110002284, 110002285, 110002286, 110002287, 110002288, 110002289, 110002290, 110002291, 110002292, 110002293, 110002294, 110002295, 110002296, 110002297, 110002298, 110002299, 110002300, 110002301, 110002302, 110002303, 110002304, 110002305, 110002306, 110002307, 110002308, 110002309, 110002310]  # List of product IDs
# # product_statuses = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # List of product statuses
# # costs = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]  # List of costs
# 
# 
# # # Convert lists to numpy arrays
# # X = np.array([product_ids, product_statuses]).T
# # y = np.array(costs)
# 
# # # Train the linear regression model
# # model = LinearRegression()
# # model.fit(X, y)
# 
# # # Specify the number of products for prediction
# # num_products = 100
# 
# # # Generate product IDs and statuses for the specified number of products
# # # predicted_product_ids = [...]  # Generate product IDs for num_products
# # # predicted_product_statuses = [...]  # Generate product statuses for num_products
# 
# # predicted_product_ids = [110002201, 110002202, 110002203, 110002204, 110002205, 110002206, 110002207, 110002208, 110002209, 110002210, 110002211, 110002212, 110002213, 110002214, 110002215, 110002216, 110002217, 110002218, 110002219, 110002220, 110002221, 110002222, 110002223, 110002224, 110002225, 110002226, 110002227, 110002228, 110002229, 110002230, 110002231, 110002232, 110002233, 110002234, 110002235, 110002236, 110002237, 110002238, 110002239, 110002240, 110002241, 110002242, 110002243, 110002244, 110002245, 110002246, 110002247, 110002248, 110002249, 110002250, 110002251, 110002252, 110002253, 110002254, 110002255, 110002256, 110002257, 110002258, 110002259, 110002260, 110002261, 110002262, 110002263, 110002264, 110002265, 110002266, 110002267, 110002268, 110002269, 110002270, 110002271, 110002272, 110002273, 110002274, 110002275, 110002276, 110002277, 110002278, 110002279, 110002280, 110002281, 110002282, 110002283, 110002284, 110002285, 110002286, 110002287, 110002288, 110002289, 110002290, 110002291, 110002292, 110002293, 110002294, 110002295, 110002296, 110002297, 110002298, 110002299, 110002300, 110002301, 110002302, 110002303, 110002304, 110002305, 110002306, 110002307, 110002308, 110002309, 110002310]  # List of product IDs
# # predicted_product_statuses = [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # List of product statuses
# 
# 
# # # Predict the cost for the specified number of products
# # predicted_costs = model.predict(np.array([predicted_product_ids, predicted_product_statuses]).T)
# 
# # # Calculate success and failure percentages
# # # Assuming you have the predicted and actual product statuses
# # # predicted_statuses = [...]  # List of predicted statuses for each product
# # # actual_statuses = [...]  # List of actual statuses for each product
# # predicted_statuses = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]  # List of product statuses
# # # actual_statuses = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1]  # List of product statuses
# # actual_statuses = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # List of product statuses
# 
# 
# # # Count successful and failed predictions
# # successful_predictions = sum(1 for pred, actual in zip(predicted_statuses, actual_statuses) if pred == actual)
# # failed_predictions = len(predicted_statuses) - successful_predictions
# 
# # # Calculate success and failure percentages
# # total_predictions = len(predicted_statuses)
# # success_percentage = (successful_predictions / total_predictions) * 100
# # failure_percentage = (failed_predictions / total_predictions) * 100
# 
# # # Output the results
# # print(f'Predicted Cost for {num_products} Products: {np.mean(predicted_costs) * num_products}')
# # print(f'Success Percentage for {num_products} Products: {success_percentage}%')
# # print(f'Failure Percentage for {num_products} Products: {failure_percentage}%')
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # import pandas as pd
# 
# # # Load your dataset from an external CSV file
# # data = pd.read_csv(r'C:\Users\Lenovo\Downloads\payload_data (1).csv')
# 
# # # Check unique values in Stage_1_Status, Stage_2_Status, and Stage_3_Status columns
# # print("Unique values in Stage_1_Status:", data['Stage_1_Status'].unique())
# # print("Unique values in Stage_2_Status:", data['Stage_2_Status'].unique())
# # print("Unique values in Stage_3_Status:", data['Stage_3_Status'].unique())
# 
# # # Initialize dictionaries to store success and failure percentages for each stage
# # stage_success = {}
# # stage_failure = {}
# 
# # # Function to calculate success and failure percentages for each stage
# # def calculate_stage_statistics(stage_column):
# #     stage_data = data[data[stage_column].notna()]  # Filter out rows with null values
# #     if not stage_data.empty:
# #         num_true = stage_data[stage_column].sum()  # Calculate the number of True values
# #         num_false = len(stage_data) - num_true     # Calculate the number of False values
# #         success_percentage = (num_true / len(stage_data)) * 100
# #         failure_percentage = 100 - success_percentage
# #         return success_percentage, failure_percentage
# #     else:
# #         return None, None
# 
# # # Calculate stage statistics for Stage 1
# # success_percentage_stage_1, failure_percentage_stage_1 = calculate_stage_statistics('Stage_1_Status')
# # if success_percentage_stage_1 is not None:
# #     stage_success['Stage 1'] = success_percentage_stage_1
# #     stage_failure['Stage 1'] = failure_percentage_stage_1
# # else:
# #     print("No data found with True or False values in 'Stage_1_Status' column.")
# 
# # # Calculate stage statistics for Stage 2
# # success_percentage_stage_2, failure_percentage_stage_2 = calculate_stage_statistics('Stage_2_Status')
# # if success_percentage_stage_2 is not None:
# #     stage_success['Stage 2'] = success_percentage_stage_2
# #     stage_failure['Stage 2'] = failure_percentage_stage_2
# # else:
# #     print("No data found with True or False values in 'Stage_2_Status' column.")
# 
# # # Calculate stage statistics for Stage 3
# # success_percentage_stage_3, failure_percentage_stage_3 = calculate_stage_statistics('Stage_3_Status')
# # if success_percentage_stage_3 is not None:
# #     stage_success['Stage 3'] = success_percentage_stage_3
# #     stage_failure['Stage 3'] = failure_percentage_stage_3
# # else:
# #     print("No data found with True or False values in 'Stage_3_Status' column.")
# 
# # # Print stage statistics
# # for stage, success_percentage in stage_success.items():
# #     print(f"{stage} success percentage:", success_percentage)
# # for stage, failure_percentage in stage_failure.items():
# #     print(f"{stage} failure percentage:", failure_percentage)
# 
# # # Calculate overall success percentage
# # overall_success_percentage = sum(stage_success.values()) / len(stage_success)
# # # Calculate overall failure percentage
# # overall_failure_percentage = 100 - overall_success_percentage
# 
# # # Calculate total cost
# # total_cost = data['Cost'].sum()
# 
# # # For prediction for 100,000 products
# # num_products = 100
# # # Predicting total cost
# # total_cost_prediction = total_cost * (num_products / len(data))
# 
# # # Predicting average success and failure percentages
# # average_success_percentage = overall_success_percentage
# # average_failure_percentage = overall_failure_percentage
# 
# # # Output predictions and stage statistics
# # print("Predicted total cost for", num_products, "products:", total_cost_prediction)
# # print("Predicted average success percentage for", num_products, "products:", average_success_percentage)
# # print("Predicted average failure percentage for", num_products, "products:", average_failure_percentage)
# 
















# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # Load the Boston housing dataset from the original source
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]

# # Convert to DataFrame if needed
# boston = pd.DataFrame(data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
# boston['PRICE'] = target

# # Selecting features and target
# X = boston.drop('PRICE', axis=1)
# y = boston['PRICE']

# # Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Feature scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Training the linear regression model
# model = LinearRegression()
# model.fit(X_train_scaled, y_train)

# # Making predictions
# y_pred = model.predict(X_test_scaled)

# # Model evaluation
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Mean Squared Error:", mse)
# print("R-squared Score:", r2)
