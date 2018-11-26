#pragma once

#ifndef TEST_HPP
#define TEST_HPP

void testMakeRandomPosition();
void testNN();
void testKifuOutput();

#ifdef USE_CATEGORICAL
void testOneHotDist();
void testDistEffect();
#endif

#endif // !TEST_HPP