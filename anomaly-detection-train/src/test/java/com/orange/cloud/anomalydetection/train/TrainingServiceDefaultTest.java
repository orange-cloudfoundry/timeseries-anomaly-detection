package com.orange.cloud.anomalydetection.train;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYDifferenceRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.junit.AfterClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringRunner;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * This example was inspired by DL4j regression examples, found here:
 * https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/regression/SingleTimestepRegressionExample.java
 *
 * It demonstrates single time step regression using LSTM
 */
@RunWith(SpringRunner.class)
@SpringBootTest(properties = {"logging.level.root=DEBUG"})
@ContextConfiguration(classes = {AnomalyDetectionTrainApplication.class}, loader = CustomSpringApplicationContextLoader.class)
public class TrainingServiceDefaultTest {

    private static final Logger LOGGER = LoggerFactory.getLogger(TrainingServiceDefaultTest.class);

    @Autowired
    TrainingService trainingService;

    private static XYSeriesCollection createSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name) {
        int nRows = data.shape()[2];
        XYSeries series = new XYSeries(name);
        for (int i = 0; i < nRows; i++) {
            series.add(i + offset, data.getDouble(i));
        }

        seriesCollection.addSeries(series);

        return seriesCollection;
    }

    /**
     * Generate an xy plot of the datasets provided.
     */
    private static void plotDataset(XYSeriesCollection c) {

        String title = "Regression example";
        String xAxisLabel = "timestamp";
        String yAxisLabel = "traffic data in bits";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = true;
        boolean urls = false;
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);

        // get a reference to the plot for further customisation...
        final XYPlot plot = chart.getXYPlot();
        plot.setDomainPannable(true);
        XYDifferenceRenderer r = new XYDifferenceRenderer(Color.green,
                Color.yellow, false);
        r.setRoundXCoordinates(true);
        plot.setDomainCrosshairLockedOnData(true);
        plot.setRangeCrosshairLockedOnData(true);
        plot.setDomainCrosshairVisible(true);
        plot.setRangeCrosshairVisible(true);
        plot.setRenderer(r);

        // Auto zoom to fit time series in initial window
        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);

        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        RefineryUtilities.centerFrameOnScreen(f);
        f.setVisible(true);
    }

    @AfterClass
    public static void tearDownClass() throws Exception {
        //ugly : prevent test from ending so that we can have a look to chart
        CountDownLatch countdown = new CountDownLatch(1);
        countdown.await(10, TimeUnit.SECONDS);
    }

    @Test
    public void should_train() throws Exception {

        File baseDir = new ClassPathResource("/").getFile();
        int miniBatchSize = 30;
        //int miniBatchSize = 1000;

        // ----- Load the training data -----
        SequenceRecordReader trainReader = new CSVSequenceRecordReader(0, ";");
        trainReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/internet-traffic-data-in-bits-fr-timestamp-train_%d.csv", 0, 0));

        //For regression, numPossibleLabels is not used. Setting it to -1 here
        DataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(trainReader, miniBatchSize, -1, 1, true);

        SequenceRecordReader testReader = new CSVSequenceRecordReader(0, ";");
        testReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/internet-traffic-data-in-bits-fr-timestamp-test_%d.csv", 0, 0));
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
        File networkConfigFile = new File("anomaly-detection-network-model.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(net, networkConfigFile, saveUpdater);

        // Now we want to save the normalizer to a binary file. For doing this, one can use the NormalizerSerializer.
        NormalizerSerializer serializer = NormalizerSerializer.getDefault();

        // Prepare a temporary file to save to and load from
        File normalizerFile = new File("anomaly-detection-data-normalizer");

        // Save the normalizer to a temporary file
        serializer.write(normalizer, normalizerFile);

        //Load the model
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(networkConfigFile);


        System.out.println("Saved and loaded parameters are equal:      " + net.params().equals(restored.params()));
        System.out.println("Saved and loaded configurations are equal:  " + net.getLayerWiseConfigurations().equals(restored.getLayerWiseConfigurations()));

        //Init rrnTimeStemp with train data and train test data
        net.rnnTimeStep(trainData.getFeatureMatrix());
        INDArray predicted_test = net.rnnTimeStep(testData.getFeatureMatrix());
        INDArray predicted_train = net.rnnTimeStep(trainData.getFeatureMatrix());

        //Revert data back to original values for plotting
        normalizer.revert(trainData);
        normalizer.revert(testData);
        normalizer.revertLabels(predicted_test);
        normalizer.revertLabels(predicted_train);

        //INDArray error = predictions.sub(labels);
        //INDArray anomalies_1 = testData.getLabels().gt(predicted_test);
        INDArray train_errors = Transforms.pow(trainData.getLabels().sub(predicted_train), 2);
        INDArray log_train_errors = Transforms.log(train_errors, 10);

        INDArray test_errors = Transforms.pow(testData.getLabels().sub(predicted_test), 2);
        INDArray log_anomalies = Transforms.log(test_errors, 10);
        //One simple task is to count the number of values that match the condition

        //ConditionBuilder builder = new ConditionBuilder();
        //builder.and(Conditions.greaterThan(0.0)).and()
        System.out.println("Original array: \n" + train_errors);

        //First, let's consider whole array reductions:
        double minValue = train_errors.minNumber().doubleValue();
        double maxValue = train_errors.maxNumber().doubleValue();
        double sum = train_errors.sumNumber().doubleValue();
        double avg = train_errors.meanNumber().doubleValue();
        double stdev = train_errors.stdNumber().doubleValue();

        System.out.println("minValue:       " + minValue);
        System.out.println("maxValue:       " + maxValue);
        System.out.println("sum:            " + sum);
        System.out.println("average:        " + avg);
        System.out.println("standard dev.:  " + stdev);
        final double upper_limit = avg + 2 * stdev;

        final INDArray upper_limit_trend = predicted_test.add(upper_limit);

        //final double lower_limit = avg - 2 * stdev;
        MatchCondition op = new MatchCondition(test_errors, Conditions.greaterThan(upper_limit));
        int countGreaterThanZero = Nd4j.getExecutioner().exec(op,Integer.MAX_VALUE).getInt(0);  //MAX_VALUE = "along all dimensions" or equivalently "for entire array"
        final Accumulation accumulation = Nd4j.getExecutioner().execAndReturn(op);
        System.out.println("Number of values predicted values greater than actual values on test dataset: " + countGreaterThanZero+"/" + testData.getFeatures().data().length());
        System.out.println("anomalies: " + accumulation);



        //Create plot with out data
        XYSeriesCollection c = new XYSeriesCollection();
        //createSeries(c, trainData.getFeatures(), 0, "Train data");
        createSeries(c, testData.getFeatures(), 0, "Actual test data");
        //createSeries(c, predicted_train, 0, "Predicted train data");
        createSeries(c, predicted_test, 0, "Predicted test data");
        //createSeries(c, upper_limit_trend, 0, "nomal limit test data");

        plotDataset(c);

        LOGGER.info("----- Example Complete -----");
    }
}
