#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

#define DEBUG_LEVEL 0
#define CREATE_VIDEO 1
#define MASTER_PROCESSOR_RANK 0
#define NUMBER_OF_PARTICLES 2
#define NUMBER_OF_STEPS 160
#define SIZE_OF_STEP 0.03
#define GRAVITATIONAL_CONSTANT 1
#define INITIAL_STATE_FILENAME "initial_state.txt"

int rank_of_processor;
int num_of_processors;

MPI_Datatype aggregateType;

double * velocities = NULL;

// Function prototypes for calculating forces, updating position & velocity and creating (not so good) images.
void calculate_forces(double masses[], double positions[], double curr_proc_forces[], int rank_of_processor, int particles_per_processor);
void update_positions_velocities(double positions[], double curr_proc_forces[], double curr_proc_velocities[], int rank_of_processor, int particles_per_processor);

// Main function.
int main(int argc, char * argv[]) {

  // Array for all positions and velocities & forces for particles belonging to current processors.
  double * masses;
  double * positions;
  double * curr_proc_velocities;
  double * curr_proc_forces;

  MPI_Init( & argc, & argv);
  MPI_Comm_size(MPI_COMM_WORLD, & num_of_processors);
  MPI_Comm_rank(MPI_COMM_WORLD, & rank_of_processor);

  // Get number of particles per processor (fancy calculation is to get the ceiling).
  int particles_per_processor = (NUMBER_OF_PARTICLES + num_of_processors - 1) / num_of_processors;

  if (rank_of_processor == MASTER_PROCESSOR_RANK) {
    velocities = malloc(2 * num_of_processors * particles_per_processor * sizeof(double));
  }

  masses = malloc(NUMBER_OF_PARTICLES * sizeof(double));
  positions = calloc(2 * num_of_processors * particles_per_processor, sizeof(double));
  curr_proc_velocities = malloc(2 * particles_per_processor * sizeof(double));
  curr_proc_forces = malloc(2 * particles_per_processor * sizeof(double));

  // This is to communicate positions and velocity of each chunk (so 4 total doubles for each particle in chunk).
  MPI_Type_contiguous(2 * particles_per_processor, MPI_DOUBLE, & aggregateType);
  MPI_Type_commit( & aggregateType);

  if (rank_of_processor == MASTER_PROCESSOR_RANK) {
    FILE * fp = fopen(INITIAL_STATE_FILENAME, "r");
    if (!fp) {
      printf("Error opening input file.\n");
      exit(1);
    }

    int counter_masses = 0;
    int counter_positions = 0;
    int counter_velocities = 0;
    int line = 0;
    for (line = 0; line < NUMBER_OF_PARTICLES; line++) {
      fscanf(fp, "%lf %lf %lf %lf %lf", & masses[counter_masses], & positions[counter_positions], & positions[counter_positions + 1], & velocities[counter_velocities], & velocities[counter_velocities + 1]);
      counter_masses += 1;
      counter_positions += 2;
      counter_velocities += 2;
    }

    fclose(fp);
  }

  MPI_Bcast(masses, NUMBER_OF_PARTICLES, MPI_DOUBLE, MASTER_PROCESSOR_RANK, MPI_COMM_WORLD);
  MPI_Bcast(positions, 2 * num_of_processors * particles_per_processor, MPI_DOUBLE, MASTER_PROCESSOR_RANK, MPI_COMM_WORLD);
  MPI_Scatter(velocities, 2 * particles_per_processor, MPI_DOUBLE, curr_proc_velocities, 2 * particles_per_processor, MPI_DOUBLE, MASTER_PROCESSOR_RANK, MPI_COMM_WORLD);

  double start_time = MPI_Wtime();

  // We simulate for the specified number of steps.
  int steps = 1;
  for (steps = 1; steps <= NUMBER_OF_STEPS; steps++) {
    calculate_forces(masses, positions, curr_proc_forces, rank_of_processor, particles_per_processor);
    update_positions_velocities(positions, curr_proc_forces, curr_proc_velocities, rank_of_processor, particles_per_processor);
    MPI_Allgather(MPI_IN_PLACE, 1, aggregateType, positions, 1, aggregateType, MPI_COMM_WORLD);
    
  }

  MPI_Gather(curr_proc_velocities, 1, aggregateType, velocities, 1, aggregateType, MASTER_PROCESSOR_RANK, MPI_COMM_WORLD);
  if (rank_of_processor == MASTER_PROCESSOR_RANK && DEBUG_LEVEL >= 1) {
    FILE * final_state = fopen("final_state.txt", "w+");
    if (!final_state) {
      printf("Error creating output file.\n");
      exit(1);
    }

    int particle = 0;
    for (particle = 0; particle < NUMBER_OF_PARTICLES; particle++) {
      fprintf(final_state, "%lf %lf %lf %lf %lf\n", masses[particle], positions[2 * particle], positions[2 * particle + 1], velocities[2 * particle], velocities[2 * particle + 1]);
    }

    fclose(final_state);
  }

  double end_time = MPI_Wtime();
  if (rank_of_processor == MASTER_PROCESSOR_RANK) {
    printf("Time elapsed = %e seconds.\n", end_time - start_time);
  }

  MPI_Type_free( & aggregateType);
  free(masses);
  free(positions);
  free(curr_proc_velocities);
  free(curr_proc_forces);
  if (rank_of_processor == MASTER_PROCESSOR_RANK) {
    free(velocities);
  }

  MPI_Finalize();
  return 0;
}

void calculate_forces(double masses[], double positions[], double curr_proc_forces[], int rank_of_processor, int particles_per_processor) {

  // Starting and ending particle for the current processor.
  int starting_index = rank_of_processor * particles_per_processor;
  int ending_index = starting_index + particles_per_processor - 1;

  if (starting_index >= NUMBER_OF_PARTICLES) {
    return;
  } else if (ending_index >= NUMBER_OF_PARTICLES) {
    ending_index = NUMBER_OF_PARTICLES - 1;
  }

  int particle = starting_index;
  for (particle = starting_index; particle <= ending_index; particle++) {
    double force_x = 0;
    double force_y = 0;
    int i = 0;
    for (i = 0; i < NUMBER_OF_PARTICLES; i++) {
      if (particle == i) {
        continue;
      }
      double x_diff = positions[2 * i] - positions[2 * particle];
      double y_diff = positions[2 * i + 1] - positions[2 * particle + 1];
      double distance = sqrt(x_diff * x_diff + y_diff * y_diff);
      double distance_cubed = distance * distance * distance;

      double force_total = GRAVITATIONAL_CONSTANT * masses[i] / distance;
      force_x += GRAVITATIONAL_CONSTANT * masses[i] * x_diff / distance_cubed;
      force_y += GRAVITATIONAL_CONSTANT * masses[i] * y_diff / distance_cubed;
    }
    curr_proc_forces[2 * (particle - starting_index)] = force_x;
    curr_proc_forces[2 * (particle - starting_index) + 1] = force_y;
    if (DEBUG_LEVEL >= 2) {
        printf("Force on particle %i = %.3f  %.3f\n", particle, force_x, force_y);
    }
  }
}

void update_positions_velocities(double positions[], double curr_proc_forces[], double curr_proc_velocities[], int rank_of_processor, int particles_per_processor) {

  // Starting and ending particle for the current processor.
  int starting_index = rank_of_processor * particles_per_processor;
  int ending_index = starting_index + particles_per_processor - 1;

  if (starting_index >= NUMBER_OF_PARTICLES) {
    return;
  } else if (ending_index >= NUMBER_OF_PARTICLES) {
    ending_index = NUMBER_OF_PARTICLES - 1;
  }

  int particle = starting_index;
  for (particle = starting_index; particle <= ending_index; particle++) {
    positions[2 * particle] += curr_proc_velocities[2 * (particle - starting_index)] * SIZE_OF_STEP + (curr_proc_forces[2 * (particle - starting_index)] * SIZE_OF_STEP * SIZE_OF_STEP / 2);
    positions[2 * particle] = fmod(positions[2 * particle], 1.00);
    positions[2 * particle + 1] += curr_proc_velocities[2 * (particle - starting_index) + 1] * SIZE_OF_STEP + (curr_proc_forces[2 * (particle - starting_index) + 1] * SIZE_OF_STEP * SIZE_OF_STEP / 2);
    positions[2 * particle + 1] = fmod(positions[2 * particle + 1], 1.00);
    curr_proc_velocities[2 * (particle - starting_index)] += curr_proc_forces[2 * (particle - starting_index)] * SIZE_OF_STEP;
    curr_proc_velocities[2 * (particle - starting_index) + 1] += curr_proc_forces[2 * (particle - starting_index) + 1] * SIZE_OF_STEP;
    if (DEBUG_LEVEL >= 2) {
        printf("Position of particle %i = %.3f  %.3f\n", particle, positions[2*particle], positions[2*particle + 1]);
    }
  }
}
