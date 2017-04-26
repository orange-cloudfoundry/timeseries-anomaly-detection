package com.orange.cloud.anomalydetection.train;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.File;

/**
 * @author Sebastien Bortolussi
 */
@Component
public class ModelTrainer implements CommandLineRunner {

    public static final String INTERNET_TRAFFIC_DATA_TRAIN = "internet-traffic-data-in-bits-fr-timestamp-train.csv";
    public static final String INTERNET_TRAFFIC_DATA_TEST = "internet-traffic-data-in-bits-fr-timestamp-test.csv";

    public static final String ANOMALY_DETECTION_NETWORK_MODEL_FILE = "anomaly-detection-network-model";
    public static final String ANOMALY_DETECTION_NETWORK_MODEL_FILE_EXT = ".zip";
    public static final String ANOMALY_DETECTION_DATA_NORMALIZER_FILE = "anomaly-detection-data-normalizer";

    @Autowired
    TrainingService trainingService;

    @Value("${build.version}")
    private String buildVersion;

    @Override
    public void run(String... args) throws Exception {

        int miniBatchSize = 30;
        //int miniBatchSize = 1000;

        // ----- Load the training data -----
        SequenceRecordReader trainReader = new CSVSequenceRecordReader(0, ";");
        trainReader.initialize(new FileSplit(new ClassPathResource(INTERNET_TRAFFIC_DATA_TRAIN).getFile()));

        //For regression, numPossibleLabels is not used. Setting it to -1 here
        DataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(trainReader, miniBatchSize, -1, 1, true);

        SequenceRecordReader testReader = new CSVSequenceRecordReader(0, ";");
        testReader.initialize(new FileSplit(new ClassPathResource(INTERNET_TRAFFIC_DATA_TEST).getFile()));
        DataSetIterator testIter = new SequenceRecordReaderDataSetIterator(testReader, miniBatchSize, -1, 1, true);

        //Create data set from iterator here since we only have a single data set
        DataSet trainData = trainIter.next();
        DataSet testData = testIter.next();

        //Normalize data, including labels (fitLabel=true)
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fitLabel(true);
        normalizer.fit(trainData);              //Collect training data statistics

        normalizer.transform(trainData);
        normalizer.transform(testData);

        final MultiLayerNetwork net = trainingService.train(trainData, testData);

        //Save the model
        File networkConfigFile = new File(getAnomalyDetectionNetworkModelFile());      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(net, networkConfigFile, saveUpdater);

        // Now we want to save the normalizer to a binary file. For doing this, one can use the NormalizerSerializer.
        NormalizerSerializer serializer = NormalizerSerializer.getDefault();

        // Prepare a temporary file to save to and load from
        File normalizerFile = new File(getAnomalyDetectionDataNormalizerFile());

        // Save the normalizer to a temporary file
        serializer.write(normalizer, normalizerFile);

    }

    private String getAnomalyDetectionDataNormalizerFile() {
        return ANOMALY_DETECTION_DATA_NORMALIZER_FILE + "_" + buildVersion;
    }

    private String getAnomalyDetectionNetworkModelFile() {
        return ANOMALY_DETECTION_NETWORK_MODEL_FILE + "_" + buildVersion + ANOMALY_DETECTION_NETWORK_MODEL_FILE_EXT;
    }
}
