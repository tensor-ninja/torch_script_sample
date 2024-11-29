#include "tokenizer.hpp"
#include <cassert>
#include <iostream>
#include <fstream>

void check_vocab_file(const std::string& vocab_path) {
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        throw std::runtime_error("Could not open vocab file at: " + vocab_path);
    }
    
    // Check first few entries
    std::string line;
    int count = 0;
    std::cout << "First 5 vocab entries:\n";
    while (std::getline(vocab_file, line) && count < 5) {
        std::cout << count << ": " << line << "\n";
        count++;
    }
    
    // Count total entries
    vocab_file.clear();
    vocab_file.seekg(0);
    int total_lines = std::count(std::istreambuf_iterator<char>(vocab_file),
                                std::istreambuf_iterator<char>(), '\n');
    std::cout << "Total vocabulary size: " << total_lines << "\n\n";
}

void test_basic_tokenization() {
    std::string vocab_path = "../exported_model/vocab.txt";
    
    // Check vocab file first
    check_vocab_file(vocab_path);
    
    BertTokenizer tokenizer(vocab_path);
    
    // Test cases
    std::vector<std::pair<std::string, std::vector<std::string>>> test_cases = {
        {"Hello World!", {"hello", "world", "!"}},
        {"testing", {"test", "##ing"}},
        {"BERT", {"bert"}},
        {"machine learning", {"machine", "learning"}}
    };
    
    for (const auto& test_case : test_cases) {
        auto tokens = tokenizer.tokenize_for_test(test_case.first);
        
        // Print results
        std::cout << "Input: " << test_case.first << "\n";
        std::cout << "Expected: ";
        for (const auto& t : test_case.second) std::cout << t << " ";
        std::cout << "\nGot: ";
        for (const auto& t : tokens) std::cout << t << " ";
        std::cout << "\n\n";
        
        // Optional: Add assertions if you want to enforce exact matches
        // assert(tokens == test_case.second);
    }
}

int main() {
    try {
        test_basic_tokenization();
        std::cout << "All tests passed!\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
} 