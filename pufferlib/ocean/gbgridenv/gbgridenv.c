#include "gbgridenv.h"

int main() {
    GbGridEnv env = {0};
    int size = GBGRID_DEFAULT_SIZE;
    env.observations = (unsigned char*)calloc(
        (size_t)GBGRID_OBS_WINDOW * (size_t)GBGRID_OBS_WINDOW,
        sizeof(unsigned char)
    );
    env.actions = (int*)calloc(1, sizeof(int));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env.agent_initial_energy = 50;

    gbgrid_alloc(&env, size, size);

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) {
                env.actions[0] = 0;
            } else if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) {
                env.actions[0] = 1;
            } else if (IsKeyDown(KEY_W) || IsKeyDown(KEY_UP)) {
                env.actions[0] = 2;
            } else if (IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN)) {
                env.actions[0] = 3;
            } else if (IsKeyDown(KEY_SPACE)) {
                env.actions[0] = 4;
            } else {
                env.actions[0] = 4;
            }
        } else {
            env.actions[0] = rand() % 5;
        }
        c_step(&env);
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    gbgrid_free(&env);
    c_close(&env);
}
