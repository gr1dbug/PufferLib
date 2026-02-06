#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

// Only use floats!
typedef struct {
    float score;
    float n; // Required as the last field
} Log;

#define GBGRID_DEFAULT_SIZE 100
#define GBGRID_OBS_WINDOW 20

typedef struct {
    float water;
    float organic_material;
    float nutr_a;
    float nutr_b;
    float nutr_c;
} Soil;

typedef enum {
    GBGRID_CELL_SOLID = 0,
    GBGRID_CELL_SOIL = 1,
    GBGRID_CELL_DEPOT = 2,
} GbGridCellType;

typedef struct {
    GbGridCellType type;
    Soil soil;
} GbGridCell;

typedef struct {
    float water;
    float organic_material;
    float nutr_a;
    float nutr_b;
    float nutr_c;
} GbGridInventory;

typedef struct {
    int x;
    int y;
    int energy;
    GbGridInventory inventory;
    float delivered_total;
} GbGridAgent;

typedef struct {
    Log log;                     // Required field
    unsigned char* observations; // Required field. Ensure type matches in .py and .c
    int* actions;                // Required field. Ensure type matches in .py and .c
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field
    int width;
    int height;
    GbGridCell* grid;
    GbGridAgent agent;
    int agent_initial_energy;
    int needs_reset;
} GbGridEnv;

static inline int gbgrid_index(const GbGridEnv* env, int x, int y) {
    if (!env || !env->grid) {
        return -1;
    }
    if (x < 0 || y < 0 || x >= env->width || y >= env->height) {
        return -1;
    }
    return y * env->width + x;
}

static inline GbGridCell* gbgrid_cell(GbGridEnv* env, int x, int y) {
    int idx = gbgrid_index(env, x, y);
    if (idx < 0) {
        return NULL;
    }
    return &env->grid[idx];
}

static inline int gbgrid_alloc(GbGridEnv* env, int width, int height) {
    if (width <= 0 || height <= 0) {
        return -1;
    }
    free(env->grid);
    env->grid = NULL;
    env->width = width;
    env->height = height;
    size_t count = (size_t)width * (size_t)height;
    env->grid = (GbGridCell*)calloc(count, sizeof(GbGridCell));
    return env->grid ? 0 : -1;
}

static inline void gbgrid_free(GbGridEnv* env) {
    free(env->grid);
    env->grid = NULL;
    env->width = 0;
    env->height = 0;
}

static inline void gbgrid_agent_reset(GbGridEnv* env) {
    if (!env) {
        return;
    }
    env->agent.energy = env->agent_initial_energy;
    env->agent.inventory.water = 0.0f;
    env->agent.inventory.organic_material = 0.0f;
    env->agent.inventory.nutr_a = 0.0f;
    env->agent.inventory.nutr_b = 0.0f;
    env->agent.inventory.nutr_c = 0.0f;
    env->agent.delivered_total = 0.0f;
    env->agent.x = 0;
    env->agent.y = 0;
    for (int y = 1; y < env->height - 1; y++) {
        for (int x = 1; x < env->width - 1; x++) {
            GbGridCell* cell = gbgrid_cell(env, x, y);
            if (cell && cell->type == GBGRID_CELL_SOIL) {
                env->agent.x = x;
                env->agent.y = y;
                return;
            }
        }
    }
    for (int y = 1; y < env->height - 1; y++) {
        for (int x = 1; x < env->width - 1; x++) {
            GbGridCell* cell = gbgrid_cell(env, x, y);
            if (cell && cell->type == GBGRID_CELL_DEPOT) {
                env->agent.x = x;
                env->agent.y = y;
                return;
            }
        }
    }
}

static inline int gbgrid_agent_move(GbGridEnv* env, GbGridAgent* agent, int x, int y) {
    if (!env || !agent) {
        return 0;
    }
    if (agent->energy <= 0) {
        return 0;
    }
    int dx = x - agent->x;
    int dy = y - agent->y;
    if (dx == 0 && dy == 0) {
        return 0;
    }
    if (dx < -1 || dx > 1 || dy < -1 || dy > 1) {
        return 0;
    }
    GbGridCell* cell = gbgrid_cell(env, x, y);
    if (!cell || (cell->type != GBGRID_CELL_SOIL && cell->type != GBGRID_CELL_DEPOT)) {
        return -1;
    }
    agent->x = x;
    agent->y = y;
    return 1;
}

static inline int gbgrid_agent_pickup(GbGridEnv* env, GbGridAgent* agent) {
    if (!env || !agent) {
        return 0;
    }

    GbGridCell* cell = gbgrid_cell(env, agent->x, agent->y);
    if (!cell || cell->type != GBGRID_CELL_SOIL) {
        return 0;
    }

    // Organic material: pick 1% to 5% of current cell content.
    float organic_pct = 0.01f + ((float)rand() / (float)RAND_MAX) * 0.04f;
    float organic_amount = cell->soil.organic_material * organic_pct;
    if (organic_amount > cell->soil.organic_material) {
        organic_amount = cell->soil.organic_material;
    }

    // Water: pick max(1% to 3% of content, 0.02), capped by available content.
    float water_pct = 0.01f + ((float)rand() / (float)RAND_MAX) * 0.02f;
    float water_amount = cell->soil.water * water_pct;
    if (water_amount < 0.02f) {
        water_amount = 0.02f;
    }
    if (water_amount > cell->soil.water) {
        water_amount = cell->soil.water;
    }

    cell->soil.organic_material -= organic_amount;
    cell->soil.water -= water_amount;
    agent->inventory.organic_material += organic_amount;
    agent->inventory.water += water_amount;

    return (organic_amount > 0.0f || water_amount > 0.0f) ? 1 : 0;
}

static inline float gbgrid_agent_deposit(GbGridEnv* env, GbGridAgent* agent) {
    if (!env || !agent) {
        return 0.0f;
    }
    GbGridCell* cell = gbgrid_cell(env, agent->x, agent->y);
    if (!cell || cell->type != GBGRID_CELL_DEPOT) {
        return 0.0f;
    }
    float deposited = 0.0f;
    deposited += agent->inventory.water;
    deposited += agent->inventory.organic_material;
    deposited += agent->inventory.nutr_a;
    deposited += agent->inventory.nutr_b;
    deposited += agent->inventory.nutr_c;

    agent->inventory.water = 0.0f;
    agent->inventory.organic_material = 0.0f;
    agent->inventory.nutr_a = 0.0f;
    agent->inventory.nutr_b = 0.0f;
    agent->inventory.nutr_c = 0.0f;
    agent->delivered_total += deposited;
    return deposited;
}

static inline void gbgrid_write_observation(GbGridEnv* env) {
    if (!env || !env->observations || env->width <= 0 || env->height <= 0) {
        return;
    }
    size_t obs_count = (size_t)GBGRID_OBS_WINDOW * (size_t)GBGRID_OBS_WINDOW;
    memset(env->observations, GBGRID_CELL_SOLID, obs_count * sizeof(unsigned char));

    int half = GBGRID_OBS_WINDOW / 2;
    int start_x = env->agent.x - half;
    int start_y = env->agent.y - half;
    for (int oy = 0; oy < GBGRID_OBS_WINDOW; oy++) {
        int y = start_y + oy;
        if (y < 0 || y >= env->height) {
            continue;
        }
        for (int ox = 0; ox < GBGRID_OBS_WINDOW; ox++) {
            int x = start_x + ox;
            if (x < 0 || x >= env->width) {
                continue;
            }
            GbGridCell* cell = gbgrid_cell(env, x, y);
            if (cell) {
                env->observations[oy * GBGRID_OBS_WINDOW + ox] = (unsigned char)cell->type;
            }
        }
    }
}

void c_reset(GbGridEnv* env) {
    if (env->grid == NULL) {
        int width = env->width > 0 ? env->width : GBGRID_DEFAULT_SIZE;
        int height = env->height > 0 ? env->height : GBGRID_DEFAULT_SIZE;
        if (gbgrid_alloc(env, width, height) != 0) {
            return;
        }
    }
    float organic_min = 0.75f;
    float organic_max = 0.9f;
    float water_min = 0.1f;
    float water_max = 0.2f;
    int interior_width = env->width - 2;
    int interior_height = env->height - 2;
    int interior_count = 0;
    if (interior_width > 0 && interior_height > 0) {
        interior_count = interior_width * interior_height;
    }
    int solids_min = 5;
    int solids_max = 25;
    int solids_count = solids_min + (rand() % (solids_max - solids_min + 1));
    if (interior_count <= 0) {
        solids_count = 0;
    } else if (solids_count > interior_count) {
        solids_count = interior_count;
    }
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            GbGridCell* cell = &env->grid[y * env->width + x];
            cell->type = GBGRID_CELL_SOLID;
            cell->soil.water = 0.0f;
            cell->soil.organic_material = 0.0f;
            cell->soil.nutr_a = 0.0f;
            cell->soil.nutr_b = 0.0f;
            cell->soil.nutr_c = 0.0f;
        }
    }
    if (interior_count > 0) {
        for (int y = 1; y < env->height - 1; y++) {
            for (int x = 1; x < env->width - 1; x++) {
                GbGridCell* cell = &env->grid[y * env->width + x];
                cell->type = GBGRID_CELL_SOIL;
                cell->soil.organic_material = organic_min +
                    ((float)rand() / (float)RAND_MAX) * (organic_max - organic_min);
                cell->soil.water = water_min +
                    ((float)rand() / (float)RAND_MAX) * (water_max - water_min);
                cell->soil.nutr_a = 0.0f;
                cell->soil.nutr_b = 0.0f;
                cell->soil.nutr_c = 0.0f;
            }
        }
    }
    if (solids_count > 0) {
        int* indices = (int*)malloc(sizeof(int) * (size_t)interior_count);
        if (indices != NULL) {
            int idx = 0;
            for (int y = 1; y < env->height - 1; y++) {
                for (int x = 1; x < env->width - 1; x++) {
                    indices[idx++] = y * env->width + x;
                }
            }
            for (int i = interior_count - 1; i > 0; i--) {
                int j = rand() % (i + 1);
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
            for (int i = 0; i < solids_count; i++) {
                GbGridCell* cell = &env->grid[indices[i]];
                cell->type = GBGRID_CELL_SOLID;
                cell->soil.water = 0.0f;
                cell->soil.organic_material = 0.0f;
                cell->soil.nutr_a = 0.0f;
                cell->soil.nutr_b = 0.0f;
                cell->soil.nutr_c = 0.0f;
            }
            free(indices);
        }
    }
    if (interior_count > 0) {
        int soil_count = 0;
        for (int y = 1; y < env->height - 1; y++) {
            for (int x = 1; x < env->width - 1; x++) {
                GbGridCell* cell = &env->grid[y * env->width + x];
                if (cell->type == GBGRID_CELL_SOIL) {
                    soil_count += 1;
                }
            }
        }
        if (soil_count > 0) {
            int depot_target = rand() % soil_count;
            int seen = 0;
            int depot_set = 0;
            for (int y = 1; y < env->height - 1; y++) {
                for (int x = 1; x < env->width - 1; x++) {
                    GbGridCell* cell = &env->grid[y * env->width + x];
                    if (cell->type != GBGRID_CELL_SOIL) {
                        continue;
                    }
                    if (seen == depot_target) {
                        cell->type = GBGRID_CELL_DEPOT;
                        cell->soil.water = 0.0f;
                        cell->soil.organic_material = 0.0f;
                        cell->soil.nutr_a = 0.0f;
                        cell->soil.nutr_b = 0.0f;
                        cell->soil.nutr_c = 0.0f;
                        depot_set = 1;
                        break;
                    }
                    seen += 1;
                }
                if (depot_set) {
                    break;
                }
            }
        }
    }
    gbgrid_agent_reset(env);
    env->needs_reset = 0;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    gbgrid_write_observation(env);
}

void c_step(GbGridEnv* env) {
    if (!env) {
        return;
    }

    if (env->needs_reset) {
        c_reset(env);
    }

    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    int moved = 0;
    int x = env->agent.x;
    int y = env->agent.y;

    // 0: left, 1: right, 2: up, 3: down, 4: no-op
    switch (env->actions[0]) {
        case 0: moved = gbgrid_agent_move(env, &env->agent, x - 1, y); break;
        case 1: moved = gbgrid_agent_move(env, &env->agent, x + 1, y); break;
        case 2: moved = gbgrid_agent_move(env, &env->agent, x, y - 1); break;
        case 3: moved = gbgrid_agent_move(env, &env->agent, x, y + 1); break;
        default: break;
    }

    /*
    if (moved < 0) {
        env->rewards[0] = -1.0f;
        env->terminals[0] = 1;
        env->log.score += env->agent.inventory.organic_material + env->agent.inventory.water;
        env->log.n += 1.0f;
        c_reset(env);
        return;
    }
    */

    int picked_up = gbgrid_agent_pickup(env, &env->agent);
    float deposited = gbgrid_agent_deposit(env, &env->agent);

    // Reward centers on useful work: deliver resources to depot.
    float reward = -0.01f;
    if (moved > 0) {
        reward += 0.01f;
    } else if (moved < 0) {
        reward -= 0.02f;
    }
    if (picked_up) {
        reward += 0.005f;
    }
    reward += deposited;
    env->rewards[0] = reward;
    if (env->rewards[0] > 1.0f) {
        env->rewards[0] = 1.0f;
    }
    if (env->rewards[0] < -1.0f) {
        env->rewards[0] = -1.0f;
    }

    if (env->agent.energy > 0) {
        env->agent.energy -= 1;
    }

    gbgrid_write_observation(env);

    if (env->agent.energy <= 0) {
        env->terminals[0] = 1;
        env->log.score += env->agent.delivered_total;
        env->log.n += 1.0f;
        env->needs_reset = 1;
        return;
    }
}

void c_render(GbGridEnv* env) {
    if (!IsWindowReady()) {
        InitWindow(1080, 720, "PufferLib GbGridEnv");
        SetTargetFPS(5);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    EndDrawing();
}

void c_close(GbGridEnv* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    gbgrid_free(env);
}
