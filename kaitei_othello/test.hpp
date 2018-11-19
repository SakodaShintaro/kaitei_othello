#pragma once

#ifndef TEST_HPP
#define TEST_HPP

void testMakeRandomPosition();
void testNN();
void testKifuOutput();

#ifdef USE_CATEGORICAL
void testOneHotDist();
#endif

#endif // !TEST_HPP