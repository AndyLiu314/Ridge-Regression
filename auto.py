import numpy as np
import q3 as q3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# load auto-mpg-regression.tsv, including  Keys are the column names, including mpg.
auto_data_all = q3.load_auto_data('auto-mpg-regression.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are q3.standard and q3.one_hot.

features1 = [('cylinders', q3.standard),
            ('displacement', q3.standard),
            ('horsepower', q3.standard),
            ('weight', q3.standard),
            ('acceleration', q3.standard),
            ('origin', q3.one_hot)]

features2 = [('cylinders', q3.one_hot),
            ('displacement', q3.standard),
            ('horsepower', q3.standard),
            ('weight', q3.standard),
            ('acceleration', q3.standard),
            ('origin', q3.one_hot)]

# Construct the standard data and label arrays
#auto_data[0] has the features for choice features1
#auto_data[1] has the features for choice features2
#The labels for both are the same, and are in auto_values
auto_data = [0, 0]
auto_values = 0
auto_data[0], auto_values = q3.auto_data_and_values(auto_data_all, features1)
auto_data[1], _ = q3.auto_data_and_values(auto_data_all, features2)

#standardize the y-values
auto_values, mu, sigma = q3.std_y(auto_values)

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------     
        
#Your code for cross-validation goes here
def run_cross_validation():
    """
    Run cross-validation for both feature sets with different polynomial orders
    and lambda values to find the best combination.
    """
    
    # Lambda values for different polynomial orders
    lambda_values_1_2 = [i * 0.01 for i in range(11)]  # 0.0, 0.01, 0.02, ..., 0.1
    lambda_values_3 = [i * 20 for i in range(11)]      # 0, 20, 40, ..., 200
    
    # Polynomial orders to test
    polynomial_orders = [1, 2, 3]
    
    # Feature sets
    feature_sets = [auto_data[0], auto_data[1]]
    feature_names = ["features1", "features2"]
    
    best_rmse = float('inf')
    best_config = None
    results = []
    
    print("Running cross-validation...")
    print("Feature Set | Poly Order | Lambda | RMSE")
    print("-" * 45)
    
    for feature_idx, (X, feature_name) in enumerate(zip(feature_sets, feature_names)):
        for poly_order in polynomial_orders:
            # Transform features to polynomial
            poly_transform = q3.make_polynomial_feature_fun(poly_order)
            X_poly = poly_transform(X)
            
            # Choose lambda values based on polynomial order
            if poly_order <= 2:
                lambda_vals = lambda_values_1_2
            else:
                lambda_vals = lambda_values_3
            
            for lam in lambda_vals:
                # Perform 10-fold cross-validation
                rmse = q3.xval_learning_alg(X_poly, auto_values, lam, 10)
                
                # Extract scalar value from numpy array if needed
                rmse_scalar = float(rmse) if hasattr(rmse, 'item') else rmse
                
                results.append({
                    'feature_set': feature_name,
                    'poly_order': poly_order,
                    'lambda': lam,
                    'rmse': rmse_scalar
                })
                
                print(f"{feature_name:11} | {poly_order:10} | {lam:6.2f} | {rmse_scalar:.6f}")
                
                # Track best configuration
                if rmse_scalar < best_rmse:
                    best_rmse = rmse_scalar
                    best_config = {
                        'feature_set': feature_name,
                        'poly_order': poly_order,
                        'lambda': lam,
                        'rmse': rmse_scalar
                    }
    
    print("\n" + "="*50)
    print("BEST CONFIGURATION:")
    print(f"Feature Set: {best_config['feature_set']}")
    print(f"Polynomial Order: {best_config['poly_order']}")
    print(f"Lambda: {best_config['lambda']}")
    print(f"Cross-validation RMSE (standardized): {best_config['rmse']:.6f}")
    
    # Convert back to original mpg units
    original_rmse = best_config['rmse'] * float(sigma)
    print(f"Cross-validation RMSE (original mpg): {original_rmse:.3f}")
    
    return best_config, results

# Run the cross-validation
if __name__ == "__main__":
    best_config, all_results = run_cross_validation()
    
    # Print summary for assignment questions
    print("\n" + "="*50)
    print("ASSIGNMENT ANSWERS:")
    print(f"3.e.1 - Best combination: {best_config['feature_set']}, polynomial order {best_config['poly_order']}, lambda = {best_config['lambda']}")
    print(f"3.e.2 - Best average cross-validation RMSE: {best_config['rmse'] * float(sigma):.3f} mpg")