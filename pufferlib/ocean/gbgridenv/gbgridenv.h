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
    if (!cell || cell->type != GBGRID_CELL_SOIL) {
        return 0;
    }
    agent->x = x;
    agent->y = y;
    agent->energy -= 1;
    if (agent->energy < 0) {
        agent->energy = 0;
    }
    return 1;
}

static inline int gbgrid_agent_pickup(GbGridEnv* env, GbGridAgent* agent) {
    (void)env;
    (void)agent;
    return 0;
}

static inline int gbgrid_agent_deposit(GbGridEnv* env, GbGridAgent* agent) {
    (void)env;
    (void)agent;
    return 0;
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
    gbgrid_agent_reset(env);
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->observations[0] = 0;
}

void c_step(GbGridEnv* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->observations[0] = 0;
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
