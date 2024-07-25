#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <iostream>

const char* words[] = {
    "I", "drink", "things", "Know", "When", "won't", "play", "out", "true", "storm", "brings", "game",
    "the", "win", "of", "enemy", "you", "wait", "thrones", "and", "or", "die", "He"
};

const int G_SIZE = sizeof(words) / sizeof(words[0]);

void printEncoded() {
    // Print header
    for (int i = 1; i <= 12; i++) {
        std::cout.width(8);
        std::cout << i << std::flush;
    }
    std::cout << '\n' << std::flush;

    // Print words
    for (int i = 0; i < G_SIZE; i++) {
        if (i > 0 && i % 12 == 0) std::cout << '\n' << std::flush;
        std::cout.width(8);
        std::cout << words[i] << std::flush;
    }
    std::cout << '\n' << std::flush;
}

#endif // PREPROCESSING_H