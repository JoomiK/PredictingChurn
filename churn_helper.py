
"""Functions to calculate churn probability, expected loss, 
calibration, and discrimination"""


def calibration(prob,outcome,n_bins=10):
    """Calibration measurement for a set of predictions.
    When predicting events at a given probability, how far is frequency
    of positive outcomes from that probability?
    NOTE: Lower scores are better
    prob: array_like, float
        Probability estimates for a set of events
    outcome: array_like, bool
        If event predicted occurred
    n_bins: int
        Number of judgement categories to prefrom calculation over.
        Prediction are binned based on probability, since "descrete" 
        probabilities aren't required. 
    """
    prob = np.array(prob)
    outcome = np.array(outcome)

    c = 0.0
    # Construct bins
    judgement_bins = np.arange(n_bins + 1) / n_bins
    # Which bin is each prediction in?
    bin_num = np.digitize(prob,judgement_bins)
    for j_bin in np.unique(bin_num):
        # Is event in bin
        in_bin = bin_num == j_bin
        # Predicted probability taken as average of preds in bin
        predicted_prob = np.mean(prob[in_bin])
        # How often did events in this bin actually happen?
        true_bin_prob = np.mean(outcome[in_bin])
        # Squared distance between predicted and true times num of obs
        c += np.sum(in_bin) * ((predicted_prob - true_bin_prob) ** 2)
    return c / len(prob)

def discrimination(prob,outcome,n_bins=10):
    """Discrimination measurement for a set of predictions.
    For each judgement category, how far from the base probability
    is the true frequency of that bin?
    NOTE: High scores are better
    prob: array_like, float
        Probability estimates for a set of events
    outcome: array_like, bool
        If event predicted occurred
    n_bins: int
        Number of judgement categories to prefrom calculation over.
        Prediction are binned based on probability, since "descrete" 
        probabilities aren't required. 
    """
    prob = np.array(prob)
    outcome = np.array(outcome)

    d = 0.0
    # Base frequency of outcomes
    base_prob = np.mean(outcome)
    # Construct bins
    judgement_bins = np.arange(n_bins + 1) / n_bins
    # Which bin is each prediction in?
    bin_num = np.digitize(prob,judgement_bins)
    for j_bin in np.unique(bin_num):
        in_bin = bin_num == j_bin
        true_bin_prob = np.mean(outcome[in_bin])
        # Squared distance between true and base times num of obs
        d += np.sum(in_bin) * ((true_bin_prob - base_prob) ** 2)
    return d / len(prob)

def print_measurements(pred_prob):
    """
    Print calibration error and discrimination
    """
    churn_prob, is_churn = pred_prob[:,1], y == 1
    print("  %-20s %.4f" % ("Calibration Error", calibration(churn_prob, is_churn)))
    print("  %-20s %.4f" % ("Discrimination", discrimination(churn_prob,is_churn)))
    print("Note -- Lower calibration is better, higher discrimination is better")


def ChurnModel(df, X, y, model):
    """
    Calculates probability of churn and expected loss, 
    and gathers customer's contact info
    """
    # Collect customer meta data
    response = df[['Area Code','Phone']]
    charges = ['Day Charge','Eve Charge','Night Charge','Intl Charge']
    response['customer_worth'] = df[charges].sum(axis=1)
    
    # Make prediction
    clf = model()
    clf = clf.fit(X,y)
    churn_prob = clf.predict_proba(X)
    response['churn_prob'] = churn_prob[:,1]
    
    # Calculate expected loss
    response['expected_loss'] = response['churn_prob'] * response['customer_worth']
    response = response.sort('expected_loss', ascending=False)
    
    # Return response DataFrame
    return response

