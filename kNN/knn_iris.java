
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.neighboursearch.*;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
public class knn_iris {

    public static void main(String[] args) {
        try {
            // Load the Iris dataset from ARFF file
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File("iris.arff"));
            Instances dataset = loader.getDataSet();
            dataset.setClassIndex(dataset.numAttributes() - 1);

            // Create the KNN classifier
            IBk classifier = new IBk();
            classifier.buildClassifier(dataset);

            // Perform classification on a test instance
            Instance testInstance = dataset.instance(0);
            double predictedClass = classifier.classifyInstance(testInstance);
            String predictedClassName = dataset.classAttribute().value((int) predictedClass);

            // Print the predicted class
            System.out.println("Predicted class: " + predictedClassName);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
