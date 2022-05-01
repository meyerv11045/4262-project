import os
import torch
import numpy as np
import pandas as pd
from model import FeatureExtractor
from datasets import DateDatasetV2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Experiment:

    def __init__(self, shape, test_name, seed = 42, folds=10):
        self.folds = folds
        self.shape = shape
        self.test_name = test_name

        torch.manual_seed(seed) # trying to make things reproducable

        if not os.path.isdir(self.test_name):
            os.mkdir(self.test_name)

    def train_classifier(self, model, optimizer, loss_fn, train_loader, val_loader, epochs, save_path):
        train_losses = [None] * epochs
        val_losses = [None] * epochs

        for epoch in range(epochs):
            model.train()
            
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for x_batch,y_batch in train_loader:
                x_batch = x_batch.float()
                predictions = model(x_batch)
            
                loss = loss_fn(predictions, y_batch.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
                
                _, class_prediction = torch.max(predictions, dim=1)
                train_correct += (class_prediction == y_batch).sum().item()
                train_total += y_batch.shape[0]
                train_loss += loss.item()
            
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            for x_batch,y_batch in val_loader:
                x_batch = x_batch.float()

                with torch.no_grad():
                    predictions = model(x_batch)
                    loss = loss_fn(predictions, y_batch.long())

                    _, class_prediction = torch.max(predictions, dim=1)
                    val_correct += (class_prediction == y_batch).sum().item()

                    val_total += y_batch.shape[0]
                    val_loss += loss.item()

            train_accuracy = train_correct / train_total
            train_loss /= train_total
            train_losses[epoch] = train_loss

            val_accuracy = val_correct / val_total
            val_loss /= val_total
            val_losses[epoch] = val_loss
            
            print(f'Epoch {epoch+1:<2} / {epochs}: Train Loss: {train_loss:.2f}  Train Accuracy: {train_accuracy*100:.2f}%  Validation Loss: {val_loss:.2f} Validaton Accuracy: {val_accuracy*100:.2f}%')
        
        torch.save(model.state_dict(), save_path)
        return model    


    def nn_classifier(self, val_data, model):
        m = len(val_data)
        y_true = np.zeros(m)
        y_pred = np.zeros(m)

        for i in range(m):
            x,y = val_data[i]
            prediction = model(torch.from_numpy(x).float())
            # get index of class w/ max prob and set its value to be y_pred[i]
            y_pred[i] = torch.argmax(prediction).item()
            y_true[i] = y

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, labels=[0,1,2,3,4,5,6], average="weighted")

        # print(f'NN Accuracy {acc * 100:.2f}%')
        # print(f'NN F1 Score: {f1:.4f}')

        return (acc, f1)


    def svm_classifier(self, X_train, y_train, X_test, y_test, kernel, c):
        svm = SVC(kernel=kernel, C=c).fit(X_train, y_train)
        pred = svm.predict(X_test)
        
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, labels=[0,1,2,3,4,5,6], average="weighted")
        
        return (acc, f1)

    
    def gen_nn_features(self, model, train_loader, test_loader, m_train, m_test):
        out_features = model.get_out_features()
        X_train = np.zeros((m_train,out_features))
        y_train = np.zeros(m_train)

        X_test = np.zeros((m_test,out_features))
        y_test = np.zeros(m_test)

        model.set_extract_feature_mode(True)

        with torch.no_grad():
            i = 0
            for x,y in train_loader:
                predictions = model(x.float()).numpy()
                
                for r in range(predictions.shape[0]):
                    X_train[i, :] = predictions[r,:]
                    y_train[i] = y[r]
                    i += 1

            i = 0
            for x,y in test_loader:
                predictions = model(x.float()).numpy()

                for r in range(predictions.shape[0]):
                    X_test[i,:] = predictions[r,:]
                    y_test[i] = y[r]
                    i += 1

        return (X_train, y_train), (X_test, y_test)


    def gen_pca_features(self, X_train, X_test, n_components):
        pca = PCA(n_components=n_components).fit(X_train)

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        return X_train, X_test


    def results_for_fold(self, fold, X_train, y_train, X_test, y_test):
        f_name = f'{self.test_name}/nn-{fold}.pt'
        model = FeatureExtractor(self.shape, torch.nn.functional.relu)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        train_data = DateDatasetV2(X_train, y_train)
        test_data = DateDatasetV2(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, pin_memory=True)

        epochs = 10

        model = self.train_classifier(model, optimizer, loss_fn, train_loader, test_loader, epochs, f_name)
    
        results = {}

        results['nn_classifier_raw_feat'] = self.nn_classifier(test_data, model)
        results['lin_svm_raw_feat'] = self.svm_classifier(X_train, y_train, X_test, y_test, kernel="linear", c=1)
        results['rbf_svm_raw_feat'] = self.svm_classifier(X_train, y_train, X_test, y_test, kernel="rbf", c=1)

        (X_train_nn, y_train_nn), (X_test_nn, y_test_nn) =  self.gen_nn_features(model, train_loader, test_loader, len(train_data), len(test_data))

        results['lin_svm_nn_feat'] = self.svm_classifier(X_train_nn, y_train_nn, X_test_nn, y_test_nn, kernel="linear", c=1)
        results['rbf_svm_nn_feat'] = self.svm_classifier(X_train_nn, y_train_nn, X_test_nn, y_test_nn, kernel="rbf", c=1)


        n_components = 20
        X_train_pca, X_test_pca = self.gen_pca_features(X_train, X_test, n_components)
        results['lin_svm_pca_feat'] = self.svm_classifier(X_train_pca, y_train, X_test_pca, y_test, kernel="linear", c=1)
        results['rbf_svm_pca_feat'] = self.svm_classifier(X_train_pca, y_train, X_test_pca, y_test, kernel="rbf", c=1)


        train_data.X = X_train_pca
        test_data.X = X_test_pca

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, pin_memory=True)
        
        
        pca_shape = self.shape.copy()
        pca_shape[0] = n_components
        pca_model = FeatureExtractor(pca_shape, torch.nn.functional.relu)
        pca_model = self.train_classifier(pca_model, optimizer, loss_fn, train_loader, test_loader, epochs, f'{self.test_name}/pca-{fold}.pt')

        results['nn_classifier_pca_feat'] = self.nn_classifier(test_data, pca_model)

        return results


    def run(self):
        data = pd.read_csv('date-data/original.csv')

        # Quantize the class labels 
        data.replace(to_replace=['BERHI', 'DEGLET', 'DOKOL', 'IRAQI', 'ROTANA', 'SAFAVI', 'SOGAY'], value=[0, 1, 2, 3, 4, 5, 6], inplace=True)

        # remove a the unneeded 1st column in the dataset
        data.drop('Unnamed: 0', axis=1, inplace=True)

        # shuffle data in-place (all samples for a class are grouped together in original dataset)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # make sure percentages of each class are equal in all folds
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=42)
        
        X= data.drop('Class', axis=1)
        y= data.Class

        results = []

        i = 1
        for train_index, test_index in skf.split(X, y):
            # Train test split for the ith fold
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            # Standardize based on train set and then apply to train and test set
            stdize = StandardScaler().fit(X_train)

            X_train = stdize.transform(X_train)
            X_test = stdize.transform(X_test)

            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            
            results.append(self.results_for_fold(i, X_train, y_train, X_test, y_test))
            i += 1
        
        self.parse_results(results)

    def parse_results(self, results):
        m = len(results)
        avg = {'nn_classifier_raw_feat': (0,0), 
                'lin_svm_raw_feat': (0,0), 
                'rbf_svm_raw_feat': (0,0), 
                'lin_svm_nn_feat': (0,0), 
                'rbf_svm_nn_feat': (0,0), 
                'lin_svm_pca_feat': (0,0), 
                'rbf_svm_pca_feat': (0,0), 
                'nn_classifier_pca_feat': (0,0)}
        
        for res in results:
            for k,v in res.items():
                avg[k] = [sum(a) for a in zip(avg[k],v)]
        
        for k,v in avg.items():
            print(f'{k:<20}: accuracy: {(v[0]/m)*100:.2f}% f1 score: {v[1]/m:.4f}')


if __name__ == '__main__':
    in_features = 34
    out_features = 34
    n_classes = 7
    shape = [in_features, 1024, 1024, out_features, n_classes]
    exp = Experiment(shape,'test2',41324)
    exp.run()