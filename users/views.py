from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
# from .models import predictions
import os
import pandas as pd
import pickle


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'dataset_phishing.csv'
    df = pd.read_csv(path, nrows=100)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

def training(request):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import preprocessing
    from sklearn.metrics import accuracy_score, confusion_matrix

    import os
    for dirname, _, filenames in os.walk('/kaggle/rajasekhar'):
        for filename in filenames:
            path = os.path.join(dirname, filename)
    path1 = settings.MEDIA_ROOT + "//" + 'dataset_phishing.csv'
    raw_dataset = pd.read_csv(path1)

    raw_dataset.describe()

    raw_dataset.shape

    pd.set_option('display.max_rows', 500)
    raw_dataset.isna().sum()
    # no missing values

    pd.reset_option('display.max_rows')

    original_dataset = raw_dataset.copy()

    class_map = {'legitimate': 0, 'phishing': 1}
    original_dataset['status'] = original_dataset['status'].map(class_map)

    # Drop the non-numeric column 'url'
    original_dataset.drop('url', axis=1, inplace=True)

    # Calculate the correlation matrix
    corr_matrix = original_dataset.corr()

    # Print the correlation matrix
    print(corr_matrix)

    plt.figure(figsize=(60,60))
    color = plt.get_cmap('viridis').copy()   # default color
    color.set_bad('lightblue') 
    sns.heatmap(corr_matrix, annot=True, linewidth=0.4, cmap=color)
    plt.savefig('heatmap')
    plt.show()

    corr_matrix.shape

    status_corr = corr_matrix['status']
    status_corr.shape

    def feature_selector_correlation(cmatrix, threshold):
        selected_features = []
        feature_score = []
        i=0
        for score in cmatrix:
            if abs(score)>threshold:
                selected_features.append(cmatrix.index[i])
                feature_score.append( ['{:3f}'.format(score)])
            i+=1
        result = list(zip(selected_features,feature_score)) 
        return result

    features_selected = feature_selector_correlation(status_corr, 0.2)
    features_selected

    selected_features = [i for (i,j) in features_selected if i != 'status']
    selected_features

    X_selected = original_dataset[selected_features]
    X_selected

    X_selected.shape

    y = original_dataset['status']

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True)

    model_random_forest = RandomForestClassifier(n_estimators=350,
                                                 random_state=42)

    model_random_forest.fit(X_train, y_train)

    def custom_accuracy_set(model, X_train, X_test, y_train, y_test, train=True):
        lb = preprocessing.LabelBinarizer()
        lb.fit(y_train)
        
        if train:
            x = X_train
            y = y_train
        elif not train:
            x = X_test
            y = y_test
            
        y_predicted = model.predict(x)
        
        accuracy = accuracy_score(y, y_predicted)
        print('model accuracy: {0:4f}'.format(accuracy))
        oconfusion_matrix = confusion_matrix(y, y_predicted)
        print('Confusion matrix: \n {}'.format(oconfusion_matrix))
        oroc_auc_score = lb.transform(y), lb.transform(y_predicted)
        
        return accuracy, oconfusion_matrix, oroc_auc_score

    # train accuracy
    train_acc = custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=True)
    train_accuracy, train_confusion_matrix, train_roc_auc_score = train_acc
    print(train_accuracy)

    # test accuracy
    test_acc = custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=False)
    test_accuracy, test_confusion_matrix, test_roc_auc_score = test_acc
    print(test_accuracy)

    import pickle

    with open('model_phishing_webpage_classifier', 'wb') as file:
        pickle.dump(model_random_forest, file)

    return render(request, 'users/training.html', {
        'train_accuracy': train_accuracy,
        'train_confusion_matrix': train_confusion_matrix,
        'train_roc_auc_score': train_roc_auc_score,
        'test_accuracy': test_accuracy,
        'test_confusion_matrix': test_confusion_matrix,
        'test_roc_auc_score': test_roc_auc_score,
    })

def prediction(request):
    if request.method == 'POST':
        # Collecting features from the request
        length_url = float(request.POST.get("length_url"))
        length_hostname = float(request.POST.get("length_hostname"))
        ip = float(request.POST.get("ip"))
        nb_dots = float(request.POST.get("nb_dots"))
        nb_qm = float(request.POST.get("nb_qm"))
        nb_eq = float(request.POST.get("nb_eq"))
        nb_slash = float(request.POST.get("nb_slash"))
        nb_www = float(request.POST.get("nb_www"))
        ratio_digits_url = float(request.POST.get("ratio_digits_url"))
        ratio_digits_host = float(request.POST.get("ratio_digits_host"))
        tld_in_subdomain = float(request.POST.get("tld_in_subdomain"))
        prefix_suffix = float(request.POST.get("prefix_suffix"))
        shortest_word_host = float(request.POST.get("shortest_word_host"))
        longest_words_raw = float(request.POST.get("longest_words_raw"))
        longest_word_path = float(request.POST.get("longest_word_path"))
        phish_hints = float(request.POST.get("phish_hints"))
        nb_hyperlinks = float(request.POST.get("nb_hyperlinks"))
        ratio_intHyperlinks = float(request.POST.get("ratio_intHyperlinks"))
        empty_title = float(request.POST.get("empty_title"))
        domain_in_title = float(request.POST.get("domain_in_title"))
        domain_age = float(request.POST.get("domain_age"))
        google_index = float(request.POST.get("google_index"))
        page_rank = float(request.POST.get("page_rank"))

        # Creating a DataFrame with the provided features
        new_data = pd.DataFrame({
            'length_url': [length_url],
            'length_hostname': [length_hostname],
            'ip': [ip],
            'nb_dots': [nb_dots],
            'nb_qm': [nb_qm],
            'nb_eq': [nb_eq],
            'nb_slash': [nb_slash],
            'nb_www': [nb_www],
            'ratio_digits_url': [ratio_digits_url],
            'ratio_digits_host': [ratio_digits_host],
            'tld_in_subdomain': [tld_in_subdomain],
            'prefix_suffix': [prefix_suffix],
            'shortest_word_host': [shortest_word_host],
            'longest_words_raw': [longest_words_raw],
            'longest_word_path': [longest_word_path],
            'phish_hints': [phish_hints],
            'nb_hyperlinks': [nb_hyperlinks],
            'ratio_intHyperlinks': [ratio_intHyperlinks],
            'empty_title': [empty_title],
            'domain_in_title': [domain_in_title],
            'domain_age': [domain_age],
            'google_index': [google_index],
            'page_rank': [page_rank]
        })

        # Load the pre-trained model
        with open('model_phishing_webpage_classifier', 'rb') as file:
            model_random_forest = pickle.load(file)

        # Make predictions
        prediction = model_random_forest.predict(new_data)
        if prediction[0] == 1:
            msg = "Phishing"
        else:
            msg = "Legitimate"

        return render(request, "users/predictForm.html", {"msg": msg})
    else:
        return render(request, 'users/predictForm.html', {})


