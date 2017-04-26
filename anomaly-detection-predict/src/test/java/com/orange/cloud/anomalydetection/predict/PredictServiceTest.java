package com.orange.cloud.anomalydetection.predict;

import org.assertj.core.api.Assertions;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

@RunWith(SpringRunner.class)
@SpringBootTest(properties = {"logging.level.root=DEBUG"})
public class PredictServiceTest {

    private static final Logger LOGGER = LoggerFactory.getLogger(PredictServiceTest.class);

    @Autowired
    private PredictionService predictionService;


    @Test
    public void predict_one() throws Exception {

        INDArray inputFeature = Nd4j.zeros(1, 1);
        inputFeature.putScalar(new int[]{0, 0}, new Double("2750.50"));

        INDArray predicted = predictionService.predict(inputFeature);

        //input : 2,750.50=> output : 3,244.10
        LOGGER.debug("input : " + inputFeature.getScalar(0) + "=> predicted : " + predicted);

        Assertions.assertThat((double) (Math.round(predicted.getDouble(0) * 100)) / 100).isEqualTo((double) 3244.10);

    }

}
