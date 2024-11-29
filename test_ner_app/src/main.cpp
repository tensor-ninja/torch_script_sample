#include "ner_model.hpp"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    try {
        NERModel ner("../exported_model/traced_model.pt",
                    "../exported_model/vocab.txt",
                    "../exported_model/label_map.json");
        
        std::vector<std::string> batch = {
        
        "Tim Cook is the CEO of Apple Inc",
        "Satya Nadella leads Microsoft Corporation",
        "Elon Musk founded SpaceX and Tesla",
        "Mark Zuckerberg created Facebook at Harvard",
        "Bill Gates started Microsoft with Paul Allen",
        "Jeff Bezos founded Amazon in Seattle",
        "Sundar Pichai works at Google in Mountain View",
        "Warren Buffett runs Berkshire Hathaway",
        "Mary Barra is the CEO of General Motors",
        "Jensen Huang founded NVIDIA Corporation",

        // Tech Companies and Locations
        "Apple headquarters is located in Cupertino California",
        "Google has offices in New York and London",
        "Amazon opened a new office in Vancouver Canada",
        "Microsoft's main campus is in Redmond Washington",
        "Meta has its headquarters in Menlo Park",
        "Intel operates facilities in Portland Oregon",
        "Twitter's headquarters is in San Francisco",
        "Adobe is based in San Jose California",
        "Oracle's headquarters is in Austin Texas",
        "Salesforce Tower dominates the San Francisco skyline",

        // Historical Figures and Places
        "Albert Einstein worked at Princeton University",
        "Leonardo da Vinci lived in Florence Italy",
        "Napoleon Bonaparte was born in Corsica France",
        "William Shakespeare wrote plays in London",
        "George Washington lived at Mount Vernon",
        "Abraham Lincoln gave the Gettysburg Address in Pennsylvania",
        "Martin Luther King Jr spoke at the Lincoln Memorial",
        "Winston Churchill led Britain during World War II",
        "Mahatma Gandhi led protests across India",
        "Nelson Mandela became president of South Africa",

        // Entertainment and Media
        "Disney acquired Pixar and Marvel Studios",
        "Netflix produces shows in Hollywood",
        "Steven Spielberg directed films for Universal Pictures",
        "Taylor Swift performed at Madison Square Garden",
        "BBC headquarters is in London England",
        "CNN is based in Atlanta Georgia",
        "Warner Bros Studios is in Burbank California",
        "Sony Pictures is located in Culver City",
        "Paramount Pictures operates in Hollywood",
        "Universal Studios has parks in Orlando Florida",

        // Sports Teams and Venues
        "The Lakers play at Crypto.com Arena in Los Angeles",
        "Manchester United plays at Old Trafford",
        "The New York Yankees play in the Bronx",
        "Real Madrid plays at Santiago Bernabeu in Spain",
        "The Chicago Bulls play at United Center",
        "Bayern Munich is based in Bavaria Germany",
        "The Boston Red Sox play at Fenway Park",
        "Paris Saint-Germain plays at Parc des Princes",
        "The Golden State Warriors play in San Francisco",
        "Liverpool FC plays at Anfield Stadium",

        // Educational Institutions
        "Harvard University is located in Cambridge Massachusetts",
        "Oxford University is in England",
        "MIT is in Cambridge near Boston",
        "Stanford University is in Palo Alto California",
        "Yale University is located in New Haven Connecticut",
        "UC Berkeley is in the San Francisco Bay Area",
        "Cambridge University is in the United Kingdom",
        "Columbia University is in New York City",
        "University of Tokyo is in Japan",
        "ETH Zurich is in Switzerland"
        };
        
        while (true) {  // Infinite loop
            // Simulate reading from a sensor
            float sensor_value = static_cast<float>(rand()) / RAND_MAX * 100.0f;
            std::cout << "Sensor reading: " << sensor_value << std::endl;
            
            // Original NER processing
            for (const auto& text : batch) {
                auto entities = ner.predict(text);
                
                std::cout << "Found entities:\n";
                for (const auto& entity : entities) {
                    std::cout << entity.first << ": " << entity.second << "\n";
                }
            }
            
            std::cout << "\n-------------------\n" << std::endl;
            
            // Sleep for 5 seconds
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
    }
    catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    return 0;
} 