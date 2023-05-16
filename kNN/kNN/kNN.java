package kNN;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Remove;

/**
 * @author Gowtham Girithar Srirangasamy
 */
public class kNN {
    public static void main(String[] args) {
        long[] execution_times = new long[10];
        long[] memory_usage = new long[10];
        for (int i = 0; i < 10; i++) {
            long start_time = System.nanoTime();
            try {
                process("IRIS.csv", 4, new int[] {0,1,2,3,4});
                //process("WineQT.csv", 10, new int[] {0,1,2,3,4,5,6,7,8,9,10});
                //process("weatherAUS.csv", -1, new int[]{2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22});
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
            long end_time = System.nanoTime();
            execution_times[i] = end_time - start_time;
            memory_usage[i] = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        }
        System.out.println("Execution times: " + Arrays.toString(execution_times));
        System.out.println("Memory usage: " + Arrays.toString(memory_usage));
    }

    /**
     * This method is to load the data set.
     *
     * @param fileName The file to be read
     * @param classIdx The zero-based index of the classifications. Set to -1 to default to last value.
     * @return the dataset as two Weka Instances. Index 0 is the training set, index 1 the testing set.
     * @throws IOException
     */
    public static Instances[] getDataSet(String fileName, int classIdx, int[] dataIndices) throws Exception {
        /**
         * we can set the file i.e., loader.setFile("finename") to load the data
         */
        //int classIdx = 1;
        /** the arffloader to load the arff file */
        CSVLoader loader = new CSVLoader();
        /** load the traing data */
        loader.setSource(new File(fileName));
        /**
         * we can also set the file like loader3.setFile(new
         * File("test-confused.arff"));
         */
        // loader.setFile(new File(fileName));
        Instances dataSet = loader.getDataSet();
        /** set the index based on the data given in the arff files */
        if (classIdx == -1)
            dataSet.setClassIndex(dataSet.numAttributes() - 1);
        else
            dataSet.setClassIndex(classIdx);

        /** Remove unwanted indices: */
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndicesArray(dataIndices);
        removeFilter.setInvertSelection(true);
        removeFilter.setInputFormat(dataSet);
        Instances filteredData = Filter.useFilter(dataSet, removeFilter);

        /** Convert nominal data: */
        NominalToBinary ntbFilter = new NominalToBinary();
        ntbFilter.setInputFormat(filteredData);

        /** Train-test split: */
        // use StratifiedRemoveFolds to randomly split the data
        StratifiedRemoveFolds filter = new StratifiedRemoveFolds();

        // set options for creating the subset of data
        String[] options = new String[6];

        options[0] = "-N";                 // indicate we want to set the number of folds
        options[1] = Integer.toString(5);  // split the data into five random folds
        options[2] = "-F";                 // indicate we want to select a specific fold
        options[3] = Integer.toString(1);  // select the first fold
        options[4] = "-S";                 // indicate we want to set the random seed
        options[5] = Integer.toString(1);  // set the random seed to 1

        filter.setOptions(options);        // set the filter options
        filter.setInputFormat(filteredData);       // prepare the filter for the data format
        filter.setInvertSelection(false);  // do not invert the selection

        // apply filter for test data here
        Instances test = Filter.useFilter(filteredData, filter);

        //  prepare and apply filter for training data here
        filter.setInvertSelection(true);     // invert the selection to get other data
        Instances train = Filter.useFilter(filteredData, filter);

        return new Instances[]{train, test};
    }

    /**
     * @param filename the file to be processed
     * @throws Exception
     */
    public static void process(String filename, int classIdx, int[] dataIndices) throws Exception {

        Instances[] data = getDataSet(filename, classIdx, dataIndices);
        if (classIdx == -1) {

            data[0].setClassIndex(data[0].numAttributes() - 1);
            data[1].setClassIndex(data[1].numAttributes() - 1);
        } else {
            data[0].setClassIndex(classIdx);
            data[1].setClassIndex(classIdx);
        }
        // k - the number of nearest neighbors to use for prediction
        Classifier ibk = new IBk(3);
        ibk.buildClassifier(data[0]);

        System.out.println(ibk);

        Evaluation eval = new Evaluation(data[0]);
        eval.evaluateModel(ibk, data[1]);
        /** Print the algorithm summary */
        System.out.println("** KNN Demo  **");
        System.out.println(eval.toSummaryString());
        //System.out.println(eval.toClassDetailsString());
        //System.out.println(eval.toMatrixString());
    }

}
