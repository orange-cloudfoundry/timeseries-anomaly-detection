package com.orange.cloud.anomalydetection.source.datacsv;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.cloud.stream.messaging.Source;
import org.springframework.messaging.support.GenericMessage;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Stream;

/**
 * @author Sebastien Bortolussi
 */
@Component
public class StreamDataFromCsvFile implements CommandLineRunner {

    private Source source;

    @Autowired
    public StreamDataFromCsvFile(Source source) {
        this.source = source;
    }

    public void run(String... strings) throws Exception {

        ClassLoader classLoader = getClass().getClassLoader();
        URI filePath = classLoader.getResource("data/internet-traffic-data-in-bits.csv").toURI();
        //read file into stream, try-with-resources
        try (Stream<String> stream = Files.lines(Paths.get(filePath))) {
            stream.forEach(observation -> {
                System.out.println("Sending value : " + observation);
                source.output().send(new GenericMessage<>(String.valueOf(observation)));
            });

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
