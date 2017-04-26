package com.orange.cloud.anomalydetection.predict;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

@SpringBootApplication
public class AnomalyDetectionPredictApplication {

    public static void main(String[] args) {
        SpringApplication.run(AnomalyDetectionPredictApplication.class, args);
    }

    @Bean
    File normalizerFile(@Value("${normalizer.path:anomaly-detection-data-normalizer}") String path) throws IOException {
        return new ClassPathResource(path).getFile();
    }

    @Bean
    File modelFile(@Value("${model.path:anomaly-detection-network-model.zip}") String path) throws IOException {
        return new ClassPathResource(path).getFile();
    }

    @Bean
    public NormalizerMinMaxScaler normalizer(File normalizerFile) throws Exception {
        // restore the normalizer from  file.
        NormalizerSerializer serializer = NormalizerSerializer.getDefault();
        return serializer.restore(normalizerFile);
    }

    @Bean
    public MultiLayerNetwork model(File modelFile) throws IOException {
        //Load the model
        return ModelSerializer.restoreMultiLayerNetwork(modelFile);
    }

}
