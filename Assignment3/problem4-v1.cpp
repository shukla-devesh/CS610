#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NSEC_SEC_MUL (1.0e9)

void gridloopsearch(
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6, double dd7, double dd8,
    double dd9, double dd10, double dd11, double dd12, double dd13, double dd14, double dd15,
    double dd16, double dd17, double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27, double dd28, double dd29,
    double dd30, double c11, double c12, double c13, double c14, double c15, double c16, double c17,
    double c18, double c19, double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29, double c210, double d2,
    double ey2, double c31, double c32, double c33, double c34, double c35, double c36, double c37,
    double c38, double c39, double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49, double c410, double d4,
    double ey4, double c51, double c52, double c53, double c54, double c55, double c56, double c57,
    double c58, double c59, double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69, double c610, double d6,
    double ey6, double c71, double c72, double c73, double c74, double c75, double c76, double c77,
    double c78, double c79, double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89, double c810, double d8,
    double ey8, double c91, double c92, double c93, double c94, double c95, double c96, double c97,
    double c98, double c99, double c910, double d9, double ey9, double c101, double c102,
    double c103, double c104, double c105, double c106, double c107, double c108, double c109,
    double c1010, double d10, double ey10, double kk);

struct timespec begin_grid, end_main;

// to store values of disp.txt
double a[120];

// to store values of grid.txt
double b[30];

int main() {
  int i, j;

  i = 0;
  FILE* fp = fopen("./disp.txt", "r");
  if (fp == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fp)) {
    if (!fscanf(fp, "%lf", &a[i])) {
      printf("Error: fscanf failed while reading disp.txt\n");
      exit(EXIT_FAILURE);
    }
    i++;
  }
  fclose(fp);

  // read grid file
  j = 0;
  FILE* fpq = fopen("./grid.txt", "r");
  if (fpq == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fpq)) {
    if (!fscanf(fpq, "%lf", &b[j])) {
      printf("Error: fscanf failed while reading grid.txt\n");
      exit(EXIT_FAILURE);
    }
    j++;
  }
  fclose(fpq);

  // grid value initialize
  // initialize value of kk;
  double kk = 0.3;

  clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);
  gridloopsearch(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12],
                 b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23], b[24],
                 b[25], b[26], b[27], b[28], b[29], a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                 a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19],
                 a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31],
                 a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43],
                 a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55],
                 a[56], a[57], a[58], a[59], a[60], a[61], a[62], a[63], a[64], a[65], a[66], a[67],
                 a[68], a[69], a[70], a[71], a[72], a[73], a[74], a[75], a[76], a[77], a[78], a[79],
                 a[80], a[81], a[82], a[83], a[84], a[85], a[86], a[87], a[88], a[89], a[90], a[91],
                 a[92], a[93], a[94], a[95], a[96], a[97], a[98], a[99], a[100], a[101], a[102],
                 a[103], a[104], a[105], a[106], a[107], a[108], a[109], a[110], a[111], a[112],
                 a[113], a[114], a[115], a[116], a[117], a[118], a[119], kk);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_main);
  printf("Total time = %f seconds\n", (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
                                          (end_main.tv_sec - begin_grid.tv_sec));

  return EXIT_SUCCESS;
}

// grid search function with loop variables

void gridloopsearch(
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6, double dd7, double dd8,
    double dd9, double dd10, double dd11, double dd12, double dd13, double dd14, double dd15,
    double dd16, double dd17, double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27, double dd28, double dd29,
    double dd30, double c11, double c12, double c13, double c14, double c15, double c16, double c17,
    double c18, double c19, double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29, double c210, double d2,
    double ey2, double c31, double c32, double c33, double c34, double c35, double c36, double c37,
    double c38, double c39, double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49, double c410, double d4,
    double ey4, double c51, double c52, double c53, double c54, double c55, double c56, double c57,
    double c58, double c59, double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69, double c610, double d6,
    double ey6, double c71, double c72, double c73, double c74, double c75, double c76, double c77,
    double c78, double c79, double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89, double c810, double d8,
    double ey8, double c91, double c92, double c93, double c94, double c95, double c96, double c97,
    double c98, double c99, double c910, double d9, double ey9, double c101, double c102,
    double c103, double c104, double c105, double c106, double c107, double c108, double c109,
    double c1010, double d10, double ey10, double kk) {
  // results values
  double x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;

  // constraint values
  double q1, q2, q3, q4, q5, q6, q7, q8, q9, q10;
  double u1, u2, u3, u4, u5, u6, u7, u8, u9, u10;

  // results points
  long pnts = 0;

  // re-calculated limits
  double e1, e2, e3, e4, e5, e6, e7, e8, e9, e10;

  // opening the "results-v0.txt" for writing he results in append mode
  FILE* fptr = fopen("./results-v1.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file !");
    exit(1);
  }

  // initialization of re calculated limits, xi's.
  e1 = kk * ey1;
  e2 = kk * ey2;
  e3 = kk * ey3;
  e4 = kk * ey4;
  e5 = kk * ey5;
  e6 = kk * ey6;
  e7 = kk * ey7;
  e8 = kk * ey8;
  e9 = kk * ey9;
  e10 = kk * ey10;

  x1 = dd1;
  x2 = dd4;
  x3 = dd7;
  x4 = dd10;
  x5 = dd13;
  x6 = dd16;
  x7 = dd19;
  x8 = dd22;
  x9 = dd25;
  x10 = dd28;

  // for loop upper values
  int s1, s2, s3, s4, s5, s6, s7, s8, s9, s10;
  s1 = floor((dd2 - dd1) / dd3);
  s2 = floor((dd5 - dd4) / dd6);
  s3 = floor((dd8 - dd7) / dd9);
  s4 = floor((dd11 - dd10) / dd12);
  s5 = floor((dd14 - dd13) / dd15);
  s6 = floor((dd17 - dd16) / dd18);
  s7 = floor((dd20 - dd19) / dd21);
  s8 = floor((dd23 - dd22) / dd24);
  s9 = floor((dd26 - dd25) / dd27);
  s10 = floor((dd29 - dd28) / dd30);

  double prev11, prev21, prev31, prev41, prev51, prev61, prev71, prev81, prev91, prev101,
         prev12, prev22, prev32, prev42, prev52, prev62, prev72, prev82, prev92, prev102, 
         prev13, prev23, prev33, prev43, prev53, prev63, prev73, prev83, prev93, prev103,
         prev14, prev24, prev34, prev44, prev54, prev64, prev74, prev84, prev94, prev104,
         prev15, prev25, prev35, prev45, prev55, prev65, prev75, prev85, prev95, prev105,
         prev16, prev26, prev36, prev46, prev56, prev66, prev76, prev86, prev96, prev106,
         prev17, prev27, prev37, prev47, prev57, prev67, prev77, prev87, prev97, prev107,
         prev18, prev28, prev38, prev48, prev58, prev68, prev78, prev88, prev98, prev108,
         prev19, prev29, prev39, prev49, prev59, prev69, prev79, prev89, prev99, prev109,
         prev110, prev210, prev310, prev410, prev510, prev610, prev710, prev810, prev910, prev1010;

  // grid search starts
  for (int r1 = 0; r1 < s1; ++r1) {
    x1 = dd1 + r1 * dd3;
    prev11 = (c11 * x1 - d1);
    prev21 = (c21 * x1 - d2);
    prev31 = (c31 * x1 - d3);
    prev41 = (c41 * x1 - d4);
    prev51 = (c51 * x1 - d5);
    prev61 = (c61 * x1 - d6);
    prev71 = (c71 * x1 - d7);
    prev81 = (c81 * x1 - d8);
    prev91 = (c91 * x1 - d9);
    prev101 = (c101 * x1 - d10);

    for (int r2 = 0; r2 < s2; ++r2) {
      x2 = dd4 + r2 * dd6;
      
      prev12 = prev11+c12*x2;
      prev22 = prev21+c22*x2;
      prev32 = prev31+c32*x2;
      prev42 = prev41+c42*x2;
      prev52 = prev51+c52*x2;
      prev62 = prev61+c62*x2;
      prev72 = prev71+c72*x2;
      prev82 = prev81+c82*x2;
      prev92 = prev91+c92*x2;
      prev102 = prev101+c102*x2;

      // prev12 = prev11 + c12 * x2;
      // prev22 = prevc22 * x2;
      // prev32 = c32 * x2;
      // prev42 = c42 * x2;
      // prev52 = c52 * x2;
      // prev62 = c62 * x2;
      // prev72 = c72 * x2;
      // prev82 = c82 * x2;
      // prev92 = c92 * x2;
      // prev102 = c102 * x2;

      // q1 += prev12;
      // q2 += prev22;
      // q3 += prev32;
      // q4 += prev42;
      // q5 += prev52;
      // q6 += prev62;
      // q7 += prev72;
      // q8 += prev82;
      // q9 += prev92;
      // q10 += prev102;

      // q1 += (c12 * x2);
      // q2 += (c22 * x2);
      // q3 += (c32 * x2);
      // q4 += (c42 * x2);
      // q5 += (c52 * x2);
      // q6 += (c62 * x2);
      // q7 += (c72 * x2);
      // q8 += (c82 * x2);
      // q9 += (c92 * x2);
      // q10 += (c102 * x2);


      for (int r3 = 0; r3 < s3; ++r3) {
        x3 = dd7 + r3 * dd9;

        prev13 = prev12+c13*x3;
        prev23 = prev22+c23*x3;
        prev33 = prev32+c33*x3;
        prev43 = prev42+c43*x3;
        prev53 = prev52+c53*x3;
        prev63 = prev62+c63*x3;
        prev73 = prev72+c73*x3;
        prev83 = prev82+c83*x3;
        prev93 = prev92+c93*x3;
        prev103 = prev102+c103*x3;

        // prev13 = c13 * x3;
        // prev23 = c23 * x3;
        // prev33 = c33 * x3;
        // prev43 = c43 * x3;
        // prev53 = c53 * x3;
        // prev63 = c63 * x3;
        // prev73 = c73 * x3;
        // prev83 = c83 * x3;
        // prev93 = c93 * x3;
        // prev103 = c103 * x3;
      
        // q1 += prev13;
        // q2 += prev23;
        // q3 += prev33;
        // q4 += prev43;
        // q5 += prev53;
        // q6 += prev63;
        // q7 += prev73;
        // q8 += prev83;
        // q9 += prev93;
        // q10 += prev103;

        // q1 += (c13 * x3);
        // q2 += (c23 * x3);
        // q3 += (c33 * x3);
        // q4 += (c43 * x3);
        // q5 += (c53 * x3);
        // q6 += (c63 * x3);
        // q7 += (c73 * x3);
        // q8 += (c83 * x3);
        // q9 += (c93 * x3);
        // q10 += (c103 * x3);
        
        for (int r4 = 0; r4 < s4; ++r4) {
          x4 = dd10 + r4 * dd12;

          prev14 = prev13+c14*x4;
          prev24 = prev23+c24*x4;
          prev34 = prev33+c34*x4;
          prev44 = prev43+c44*x4;
          prev54 = prev53+c54*x4;
          prev64 = prev63+c64*x4;
          prev74 = prev73+c74*x4;
          prev84 = prev83+c84*x4;
          prev94 = prev93+c94*x4;
          prev104 = prev103+c104*x4;

          // prev14 = c14 * x4;
          // prev24 = c24 * x4;
          // prev34 = c34 * x4;
          // prev44 = c44 * x4;
          // prev54 = c54 * x4;
          // prev64 = c64 * x4;
          // prev74 = c74 * x4;
          // prev84 = c84 * x4;
          // prev94 = c94 * x4;
          // prev104 = c104 * x4;
      
          // q1 += prev14;
          // q2 += prev24;
          // q3 += prev34;
          // q4 += prev44;
          // q5 += prev54;
          // q6 += prev64;
          // q7 += prev74;
          // q8 += prev84;
          // q9 += prev94;
          // q10 += prev104;

          // q1 += (c14 * x4);
          // q2 += (c24 * x4);
          // q3 += (c34 * x4);
          // q4 += (c44 * x4);
          // q5 += (c54 * x4);
          // q6 += (c64 * x4);
          // q7 += (c74 * x4);
          // q8 += (c84 * x4);
          // q9 += (c94 * x4);
          // q10 += (c104 * x4);

          for (int r5 = 0; r5 < s5; ++r5) {
            x5 = dd13 + r5 * dd15;

            // prev15 = c15 * x5;
            // prev25 = c25 * x5;
            // prev35 = c35 * x5;
            // prev45 = c45 * x5;
            // prev55 = c55 * x5;
            // prev65 = c65 * x5;
            // prev75 = c75 * x5;
            // prev85 = c85 * x5;
            // prev95 = c95 * x5;
            // prev105 = c105 * x5;
            
            prev15 = prev14+c15*x5;
            prev25 = prev24+c25*x5;
            prev35 = prev34+c35*x5;
            prev45 = prev44+c45*x5;
            prev55 = prev54+c55*x5;
            prev65 = prev64+c65*x5;
            prev75 = prev74+c75*x5;
            prev85 = prev84+c85*x5;
            prev95 = prev94+c95*x5;
            prev105 = prev104+c105*x5;

            // q1 += prev15;
            // q2 += prev25;
            // q3 += prev35;
            // q4 += prev45;
            // q5 += prev55;
            // q6 += prev65;
            // q7 += prev75;
            // q8 += prev85;
            // q9 += prev95;
            // q10 += prev105;

            // q1 += (c15 * x5);
            // q2 += (c25 * x5);
            // q3 += (c35 * x5);
            // q4 += (c45 * x5);
            // q5 += (c55 * x5);
            // q6 += (c65 * x5);
            // q7 += (c75 * x5);
            // q8 += (c85 * x5);
            // q9 += (c95 * x5);
            // q10 += (c105 * x5);

            for (int r6 = 0; r6 < s6; ++r6) {
              x6 = dd16 + r6 * dd18;

            // prev16 = c16  * x6;
            // prev26 = c26 * x6;
            // prev36 = c36 * x6;
            // prev46 = c46 * x6;
            // prev56 = c56 * x6;
            // prev66 = c66 * x6;
            // prev76 = c76 * x6;
            // prev86 = c86 * x6;
            // prev96 = c96 * x6;
            // prev106 = c106 * x6;

              prev16 = prev15+c16*x6;
              prev26 = prev25+c26*x6;
              prev36 = prev35+c36*x6;
              prev46 = prev45+c46*x6;
              prev56 = prev55+c56*x6;
              prev66 = prev65+c66*x6;
              prev76 = prev75+c76*x6;
              prev86 = prev85+c86*x6;
              prev96 = prev95+c96*x6;
              prev106 = prev105+c106*x6;

            // q1 += prev16;
            // q2 += prev26;
            // q3 += prev36;
            // q4 += prev46;
            // q5 += prev56;
            // q6 += prev66;
            // q7 += prev76;
            // q8 += prev86;
            // q9 += prev96;
            // q10 += prev106;

              // q1 += (c16 * x6);
              // q2 += (c26 * x6);
              // q3 += (c36 * x6);
              // q4 += (c46 * x6);
              // q5 += (c56 * x6);
              // q6 += (c66 * x6);
              // q7 += (c76 * x6);
              // q8 += (c86 * x6);
              // q9 += (c96 * x6);
              // q10 += (c106 * x6);

              for (int r7 = 0; r7 < s7; ++r7) {
                x7 = dd19 + r7 * dd21;

                // prev17 = c17 * x7;
                // prev27 = c27 * x7;
                // prev37 = c37 * x7;
                // prev47 = c47 * x7;
                // prev57 = c57 * x7;
                // prev67 = c67 * x7;
                // prev77 = c77 * x7;
                // prev87 = c87 * x7;
                // prev97 = c97 * x7;
                // prev107 = c107 * x7;

                prev17 = prev16+c17*x7;
                prev27 = prev26+c27*x7;
                prev37 = prev36+c37*x7;
                prev47 = prev46+c47*x7;
                prev57 = prev56+c57*x7;
                prev67 = prev66+c67*x7;
                prev77 = prev76+c77*x7;
                prev87 = prev86+c87*x7;
                prev97 = prev96+c97*x7;
                prev107 = prev106+c107*x7;

                // q1 += prev17;
                // q2 += prev27;
                // q3 += prev37;
                // q4 += prev47;
                // q5 += prev57;
                // q6 += prev67;
                // q7 += prev77;
                // q8 += prev87;
                // q9 += prev97;
                // q10 += prev107;

                // q1 += (c17 * x7);
                // q2 += (c27 * x7);
                // q3 += (c37 * x7);
                // q4 += (c47 * x7);
                // q5 += (c57 * x7);
                // q6 += (c67 * x7);
                // q7 += (c77 * x7);
                // q8 += (c87 * x7);
                // q9 += (c97 * x7);
                // q10 += (c107 * x7);

                for (int r8 = 0; r8 < s8; ++r8) {
                  x8 = dd22 + r8 * dd24;

                  // prev18 = c18 * x8;
                  // prev28 = c28 * x8;
                  // prev38 = c38 * x8;
                  // prev48 = c48 * x8;
                  // prev58 = c58 * x8;
                  // prev68 = c68 * x8;
                  // prev78 = c78 * x8;
                  // prev88 = c88 * x8;
                  // prev98 = c98 * x8;
                  // prev108 = c108 * x8;

                  prev18 = prev17+c18*x8;
                  prev28 = prev27+c28*x8;
                  prev38 = prev37+c38*x8;
                  prev48 = prev47+c48*x8;
                  prev58 = prev57+c58*x8;
                  prev68 = prev67+c68*x8;
                  prev78 = prev77+c78*x8;
                  prev88 = prev87+c88*x8;
                  prev98 = prev97+c98*x8;
                  prev108 = prev107+c108*x8;

                  // q1 += prev18;
                  // q2 += prev28;
                  // q3 += prev38;
                  // q4 += prev48;
                  // q5 += prev58;
                  // q6 += prev68;
                  // q7 += prev78;
                  // q8 += prev88;
                  // q9 += prev98;
                  // q10 += prev108;

                  // q1 += (c18 * x8);
                  // q2 += (c28 * x8);
                  // q3 += (c38 * x8);
                  // q4 += (c48 * x8);
                  // q5 += (c58 * x8);
                  // q6 += (c68 * x8);
                  // q7 += (c78 * x8);
                  // q8 += (c88 * x8);
                  // q9 += (c98 * x8);
                  // q10 += (c108 * x8);

                  for (int r9 = 0; r9 < s9; ++r9) {
                    x9 = dd25 + r9 * dd27;

                    // prev19 = c19 * x9;
                    // prev29 = c29 * x9;
                    // prev39 = c39 * x9;
                    // prev49 = c49 * x9;
                    // prev59 = c59 * x9;
                    // prev69 = c69 * x9;
                    // prev79 = c79 * x9;
                    // prev89 = c89 * x9;
                    // prev99 = c99 * x9;
                    // prev109 = c109 * x9;

                    prev19 = prev18+c19*x9;
                    prev29 = prev28+c29*x9;
                    prev39 = prev38+c39*x9;
                    prev49 = prev48+c49*x9; 
                    prev59 = prev58+c59*x9;
                    prev69 = prev68+c69*x9;
                    prev79 = prev78+c79*x9;
                    prev89 = prev88+c89*x9;
                    prev99 = prev98+c99*x9;
                    prev109 = prev108+c109*x9;

                    // q1 += prev19;
                    // q2 += prev29;
                    // q3 += prev39;
                    // q4 += prev49;
                    // q5 += prev59;
                    // q6 += prev69;
                    // q7 += prev79;
                    // q8 += prev89;
                    // q9 += prev99;
                    // q10 += prev109;

                    // q1 += (c19 * x9);
                    // q2 += (c29 * x9);
                    // q3 += (c39 * x9);
                    // q4 += (c49 * x9);
                    // q5 += (c59 * x9);
                    // q6 += (c69 * x9);
                    // q7 += (c79 * x9);
                    // q8 += (c89 * x9);
                    // q9 += (c99 * x9);
                    // q10 += (c109 * x9);

                    for (int r10 = 0; r10 < s10; ++r10) {
                      x10 = dd28 + r10 * dd30;

                      // constraints

                      // prev110 = c110 * x10;
                      // prev210 = c210 * x10;
                      // prev310 = c310 * x10;
                      // prev410 = c410 * x10;
                      // prev510 = c510 * x10;
                      // prev610 = c610 * x10;
                      // prev710 = c710 * x10;
                      // prev810 = c810 * x10;
                      // prev910 = c910 * x10;
                      // prev1010 = c1010 * x10;

                      prev110 = prev19+c110*x10;
                      prev210 = prev29+c210*x10;
                      prev310 = prev39+c310*x10;
                      prev410 = prev49+c410*x10;
                      prev510 = prev59+c510*x10;
                      prev610 = prev69+c610*x10;
                      prev710 = prev79+c710*x10;
                      prev810 = prev89+c810*x10;
                      prev910 = prev99+c910*x10;
                      prev1010 = prev109+c1010*x10;

                      // q1 = prev11 + prev12 + prev13 + prev14 + prev15 + prev16 + prev17 + prev18 + prev19 + prev110;
                      // q1 = prev11 + prev12 + prev13 + prev14 + prev15 + prev16 + prev17 + prev18 + prev19 + prev110;


                      // q1 = prev11 + prev12 + prev13 + prev14 + prev15 + prev16 + prev17 + prev18 + prev19 + prev110;
                      // q1 = prev11 + prev12 + prev13 + prev14 + prev15 + prev16 + prev17 + prev18 + prev19 + prev110;


                      // q1 = prev11+prev12+prev13+prev14+prev15+prev16+prev17+prev18+prev19+prev110;
                      // q2 = prev21+prev22+prev23+prev24+prev25+prev26+prev27+prev28+prev29+prev210;
                      // q3 = prev31+prev32+prev33+prev34+prev35+prev36+prev37+prev38+prev39+prev310;
                      // q4 = prev41+prev42+prev43+prev44+prev45+prev46+prev47+prev48+prev49+prev410;
                      // q5 = prev51+prev52+prev53+prev54+prev55+prev56+prev57+prev58+prev59+prev510;
                      // q6 = prev61+prev62+prev63+prev64+prev65+prev66+prev67+prev68+prev69+prev610;
                      // q7 = prev71+prev72+prev73+prev74+prev75+prev76+prev77+prev78+prev79+prev710;
                      // q8 = prev81+prev82+prev83+prev84+prev85+prev86+prev87+prev88+prev89+prev810;
                      // q9 = prev91+prev92+prev93+prev94+prev95+prev96+prev97+prev98+prev99+prev910;
                      // q10 = prev101+prev102+prev103+prev104+prev105+prev106+prev107+prev108+prev109+prev1010;

                      // q1 += prev110;
                      // q2 += prev210;
                      // q3 += prev310;
                      // q4 += prev410;
                      // q5 += prev510;
                      // q6 += prev610;
                      // q7 += prev710;
                      // q8 += prev810;
                      // q9 += prev910;
                      // q10 += prev1010;

                      // q1 += c110 * x10;
                      // q2 += c210 * x10;
                      // q3 += c310 * x10;
                      // q4 += c410 * x10;
                      // q5 += c510 * x10;
                      // q6 += c610 * x10;
                      // q7 += c710 * x10;
                      // q8 += c810 * x10;
                      // q9 += c910 * x10;
                      // q10 += c1010 * x10;
                      
                      // u1 = fabs(q1);
                      // u2 = fabs(q2);
                      // u3 = fabs(q3);
                      // u4 = fabs(q4);
                      // u5 = fabs(q5);
                      // u6 = fabs(q6);
                      // u7 = fabs(q7);
                      // u8 = fabs(q8);
                      // u9 = fabs(q9);
                      // u10 = fabs(q10);

                      u1 = fabs(prev110);
                      u2 = fabs(prev210);
                      u3 = fabs(prev310);
                      u4 = fabs(prev410);
                      u5 = fabs(prev510);
                      u6 = fabs(prev610);
                      u7 = fabs(prev710);
                      u8 = fabs(prev810);
                      u9 = fabs(prev910);
                      u10 = fabs(prev1010);

                      // q1 = fabs(q1 + c110 * x10);
                      // q2 = fabs(q2 + c210 * x10);
                      // q3 = fabs(q3 + c310 * x10);
                      // q4 = fabs(q4 + c410 * x10);
                      // q5 = fabs(q5 + c510 * x10);
                      // q6 = fabs(q6 + c610 * x10);
                      // q7 = fabs(q7 + c710 * x10);
                      // q8 = fabs(q8 + c810 * x10);
                      // q9 = fabs(q9 + c910 * x10);
                      // q10 = fabs(q10 + c1010 * x10);
                      // printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", q1, e1, q2, e2, q3, e3, q4, e4, q5, e5, 
                      // q6, e6, q7, e7, q8, e8, q9, e9, q10, e10);
                      if ((u1 <= e1) && (u2 <= e2) && (u3 <= e3) && (u4 <= e4) && (u5 <= e5) &&
                          (u6 <= e6) && (u7 <= e7) && (u8 <= e8) && (u9 <= e9) && (u10 <= e10)) {
                        pnts = pnts + 1;
                        // xi's which satisfy the constraints to be written in file
                        fprintf(fptr, "%lf\t", x1);
                        fprintf(fptr, "%lf\t", x2);
                        fprintf(fptr, "%lf\t", x3);
                        fprintf(fptr, "%lf\t", x4);
                        fprintf(fptr, "%lf\t", x5);
                        fprintf(fptr, "%lf\t", x6);
                        fprintf(fptr, "%lf\t", x7);
                        fprintf(fptr, "%lf\t", x8);
                        fprintf(fptr, "%lf\t", x9);
                        fprintf(fptr, "%lf\n", x10);
                      }
                      // q1 -= prev110;
                      // q2 -= prev210;
                      // q3 -= prev310;
                      // q4 -= prev410;
                      // q5 -= prev510;
                      // q6 -= prev610;
                      // q7 -= prev710;
                      // q8 -= prev810;
                      // q9 -= prev910;
                      // q10 -= prev1010;

                      // q1 -= c110 * x10;
                      // q2 -= c210 * x10;
                      // q3 -= c310 * x10;
                      // q4 -= c410 * x10;
                      // q5 -= c510 * x10;
                      // q6 -= c610 * x10;
                      // q7 -= c710 * x10;
                      // q8 -= c810 * x10;
                      // q9 -= c910 * x10;
                      // q10 -= c1010 * x10;
                    }
                    // q1 -= prev19;
                    // q2 -= prev29;
                    // q3 -= prev39;
                    // q4 -= prev49;
                    // q5 -= prev59;
                    // q6 -= prev69;
                    // q7 -= prev79;
                    // q8 -= prev89;
                    // q9 -= prev99;
                    // q10 -= prev109;

                    // q1 -= (c19 * x9);
                    // q2 -= (c29 * x9);
                    // q3 -= (c39 * x9);
                    // q4 -= (c49 * x9);
                    // q5 -= (c59 * x9);
                    // q6 -= (c69 * x9);
                    // q7 -= (c79 * x9);
                    // q8 -= (c89 * x9);
                    // q9 -= (c99 * x9);
                    // q10 -= (c109 * x9);
                  }
                  // q1 -= prev18;
                  // q2 -= prev28;
                  // q3 -= prev38;
                  // q4 -= prev48;
                  // q5 -= prev58;
                  // q6 -= prev68;
                  // q7 -= prev78;
                  // q8 -= prev88;
                  // q9 -= prev98;
                  // q10 -= prev108;

                  // q1 -= (c18 * x8);
                  // q2 -= (c28 * x8);
                  // q3 -= (c38 * x8);
                  // q4 -= (c48 * x8);
                  // q5 -= (c58 * x8);
                  // q6 -= (c68 * x8);
                  // q7 -= (c78 * x8);
                  // q8 -= (c88 * x8);
                  // q9 -= (c98 * x8);
                  // q10 -= (c108 * x8);
                }
                // q1 -= prev17;
                // q2 -= prev27;
                // q3 -= prev37;
                // q4 -= prev47;
                // q5 -= prev57;
                // q6 -= prev67;
                // q7 -= prev77;
                // q8 -= prev87;
                // q9 -= prev97;
                // q10 -= prev107;

                // q1 -= (c17 * x7);
                // q2 -= (c27 * x7);
                // q3 -= (c37 * x7);
                // q4 -= (c47 * x7);
                // q5 -= (c57 * x7);
                // q6 -= (c67 * x7);
                // q7 -= (c77 * x7);
                // q8 -= (c87 * x7);
                // q9 -= (c97 * x7);
                // q10 -= (c107 * x7);
              }
              // q1 -= prev16;
              // q2 -= prev26;
              // q3 -= prev36;
              // q4 -= prev46;
              // q5 -= prev56;
              // q6 -= prev66;
              // q7 -= prev76;
              // q8 -= prev86;
              // q9 -= prev96;
              // q10 -= prev106;

              // q1 -= (c16 * x6);
              // q2 -= (c26 * x6);
              // q3 -= (c36 * x6);
              // q4 -= (c46 * x6);
              // q5 -= (c56 * x6);
              // q6 -= (c66 * x6);
              // q7 -= (c76 * x6);
              // q8 -= (c86 * x6);
              // q9 -= (c96 * x6);
              // q10 -= (c106 * x6);
            }
            // q1 -= prev15;
            // q2 -= prev25;
            // q3 -= prev35;
            // q4 -= prev45;
            // q5 -= prev55;
            // q6 -= prev65;
            // q7 -= prev75;
            // q8 -= prev85;
            // q9 -= prev95;
            // q10 -= prev105;

            // q1 -= (c15 * x5);
            // q2 -= (c25 * x5);
            // q3 -= (c35 * x5);
            // q4 -= (c45 * x5);
            // q5 -= (c55 * x5);
            // q6 -= (c65 * x5);
            // q7 -= (c75 * x5);
            // q8 -= (c85 * x5);
            // q9 -= (c95 * x5);
            // q10 -= (c105 * x5);
            
          }
          // q1 -= prev14;
          // q2 -= prev24;
          // q3 -= prev34;
          // q4 -= prev44;
          // q5 -= prev54;
          // q6 -= prev64;
          // q7 -= prev74;
          // q8 -= prev84;
          // q9 -= prev94;
          // q10 -= prev104;

          // q1 -= (c14 * x4);
          // q2 -= (c24 * x4);
          // q3 -= (c34 * x4);
          // q4 -= (c44 * x4);
          // q5 -= (c54 * x4);
          // q6 -= (c64 * x4);
          // q7 -= (c74 * x4);
          // q8 -= (c84 * x4);
          // q9 -= (c94 * x4);
          // q10 -= (c104 * x4);
        }
        // q1 -= prev13;
        // q2 -= prev23;
        // q3 -= prev33;
        // q4 -= prev43;
        // q5 -= prev53;
        // q6 -= prev63;
        // q7 -= prev73;
        // q8 -= prev83;
        // q9 -= prev93;
        // q10 -= prev103;

        // q1 -= (c13 * x3);
        // q2 -= (c23 * x3);
        // q3 -= (c33 * x3);
        // q4 -= (c43 * x3);
        // q5 -= (c53 * x3);
        // q6 -= (c63 * x3);
        // q7 -= (c73 * x3);
        // q8 -= (c83 * x3);
        // q9 -= (c93 * x3);
        // q10 -= (c103 * x3);
      }
      // q1 -= prev12;
      // q2 -= prev22;
      // q3 -= prev32;
      // q4 -= prev42;
      // q5 -= prev52;
      // q6 -= prev62;
      // q7 -= prev72;
      // q8 -= prev82;
      // q9 -= prev92;
      // q10 -= prev102;

      // q1 -= (c12 * x2);
      // q2 -= (c22 * x2);
      // q3 -= (c32 * x2);
      // q4 -= (c42 * x2);
      // q5 -= (c52 * x2);
      // q6 -= (c62 * x2);
      // q7 -= (c72 * x2);
      // q8 -= (c82 * x2);
      // q9 -= (c92 * x2);
      // q10 -= (c102 * x2);
    }
  }

  fclose(fptr);
  printf("result pnts: %ld\n", pnts);

  // end function gridloopsearch
}
