#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <lbm/communications.hpp>
#include <lbm/physics.hpp>
#include <lbm/tpl_loader.hpp>

/// @brief Saves the result of one step of computation.
///
/// This function can be called multiple times when a MPI save on multiple
/// processes happens (e.g. saving them one at a time on each domain).
/// Writes only velocities and macroscopic densities in the form of single
/// precision floating-point numbers.
///
/// @param fp File descriptor to write to.
/// @param mesh Domain to save.
void save_frame(FILE *fp, const Mesh *mesh)
{
    // Write buffer to write float instead of double
    lbm_file_entry_t buffer[WRITE_BUFFER_ENTRIES];
    // Loop on all values
    size_t cnt = 0;
    for (size_t i = 1; i < mesh->width - 1; i++)
    {
        for (size_t j = 1; j < mesh->height - 1; j++)
        {
            // Compute macroscopic values
            const double density = get_cell_density(Mesh_get_cell(mesh, i, j));
            Vector v;
            get_cell_velocity(v, Mesh_get_cell(mesh, i, j), density);
            const double norm = std::sqrt(get_vect_norm_2(v, v));
            // Fill buffer
            buffer[cnt].rho = density;
            buffer[cnt].v = norm;
            cnt++;
            assert(cnt <= WRITE_BUFFER_ENTRIES);
            // Flush buffer if full
            if (cnt == WRITE_BUFFER_ENTRIES)
            {
                fwrite(buffer, sizeof(lbm_file_entry_t), cnt, fp);
                cnt = 0;
            }
        }
    }
    // Final flush
    if (cnt != 0)
    {
        fwrite(buffer, sizeof(lbm_file_entry_t), cnt, fp);
    }
}
static int lbm_helper_pgcd(int a, int b)
{
    int c;
    while (b != 0)
    {
        c = a % b;
        a = b;
        b = c;
    }
    return a;
}
static int PMPI_Syncall_cb(MPI_Comm comm)
{
    static int (*__builtin_fence_ps)() = rt_tpl_sync(comm, __builtin_fence_ps, MPI_HINT_VTBL);
    return __builtin_fence_ps();
}
static int helper_get_rank_id(int nb_x, int nb_y, int rank_x, int rank_y)
{
    if (rank_x < 0 || rank_x >= nb_x)
    {
        return -1;
    }
    else if (rank_y < 0 || rank_y >= nb_y)
    {
        return -1;
    }
    else
    {
        return (rank_x + rank_y * nb_x);
    }
}

void lbm_comm_print(const lbm_comm_t *mesh_comm)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    static bool first_call = true;
    if (first_call && rank == RANK_MASTER)
    {
        first_call = false;
        fprintf(stderr, "%4s| %8s %8s %8s %8s | %12s %12s %12s %12s | %6s %6s | %6s %6s\n", "RANK", "TOP", "BOTTOM",
                "LEFT", "RIGHT", "TOP LEFT", "TOP RIGHT", "BOTTOM LEFT", "BOTTOM RIGHT", "POS X", "POS Y", "DIM X",
                "DIM Y");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    fprintf(stderr,
            "%4d| %7d  %7d  %7d  %7d  | %11d  %11d  %11d  %11d  | %5d  %5d  | "
            "%5d  %5d \n",
            rank, mesh_comm->top_id, mesh_comm->bottom_id, mesh_comm->left_id, mesh_comm->right_id,
            mesh_comm->corner_id[CORNER_TOP_LEFT], mesh_comm->corner_id[CORNER_TOP_RIGHT],
            mesh_comm->corner_id[CORNER_BOTTOM_LEFT], mesh_comm->corner_id[CORNER_BOTTOM_RIGHT], mesh_comm->x,
            mesh_comm->y, mesh_comm->width, mesh_comm->height);
}

void lbm_comm_init(lbm_comm_t *mesh_comm, int rank, int comm_size, uint32_t width, uint32_t height, uint32_t nb_x,
                   uint32_t nb_y)
{

    if (comm_size == 1) {
        mesh_comm->nb_x = 1;
        mesh_comm->nb_y = 1;
        mesh_comm->width = width + 2;
        mesh_comm->height = height + 2;
        mesh_comm->x = 0;
        mesh_comm->y = 0;
        
        // Pas de voisins
        mesh_comm->left_id = -1;
        mesh_comm->right_id = -1;
        mesh_comm->top_id = -1;
        mesh_comm->bottom_id = -1;
        for (int i = 0; i < 4; i++) {
            mesh_comm->corner_id[i] = -1;
        }
        
        // Pas de buffer de communication
        mesh_comm->buffer = NULL;
        
        // Topologie MPI
        mesh_comm->graph_comm = MPI_COMM_SELF;
        mesh_comm->mpi_degree = 0;
        
        // INITIALISATION IMPORTANTE : map à -1 pour toutes les directions
        for (int i = 0; i < 8; i++) {
            mesh_comm->map[i] = -1;
        }
        
        // Buffers vides
        mesh_comm->sendbuf = NULL;
        mesh_comm->recvbuf = NULL;
        mesh_comm->buf_size = 0;
        
        // Initialiser les voisins (tous à -1)
        for (int i = 0; i < 8; i++) {
            mesh_comm->neighbors[i] = -1;
        }
        
        lbm_comm_print(mesh_comm);
        return;
    }
    
    // DÉTERMINATION AUTOMATIQUE SI nb_x = nb_y = 1 ou les tailles données sont pas correctes
    if ((nb_x == 1 && nb_y == 1) || nb_x * nb_y != comm_size) {
        int nb_y_calc = lbm_helper_pgcd(comm_size, width);
        int nb_x_calc = comm_size / nb_y_calc;
        nb_x = nb_x_calc;
        nb_y = nb_y_calc;
    }

    if (height % nb_y != 0 || width % nb_x != 0)
    {
        fatal("Can't get a 2D cut for current problem size and number of processes.");
    }

    // Compute current rank position (ID)
    int rank_x = rank % nb_x;
    int rank_y = rank / nb_x;

    // Setup nb
    mesh_comm->nb_x = nb_x;
    mesh_comm->nb_y = nb_y;

    // Setup size (+2 for ghost cells on border)
    mesh_comm->width = width / nb_x + 2;
    mesh_comm->height = height / nb_y + 2;

    // Setup position
    mesh_comm->x = rank_x * width / nb_x;
    mesh_comm->y = rank_y * height / nb_y;

    // Compute neighbour nodes id
    mesh_comm->left_id = helper_get_rank_id(nb_x, nb_y, rank_x - 1, rank_y);
    mesh_comm->right_id = helper_get_rank_id(nb_x, nb_y, rank_x + 1, rank_y);
    mesh_comm->top_id = helper_get_rank_id(nb_x, nb_y, rank_x, rank_y - 1);
    mesh_comm->bottom_id = helper_get_rank_id(nb_x, nb_y, rank_x, rank_y + 1);
    mesh_comm->corner_id[CORNER_TOP_LEFT] = helper_get_rank_id(nb_x, nb_y, rank_x - 1, rank_y - 1);
    mesh_comm->corner_id[CORNER_TOP_RIGHT] = helper_get_rank_id(nb_x, nb_y, rank_x + 1, rank_y - 1);
    mesh_comm->corner_id[CORNER_BOTTOM_LEFT] = helper_get_rank_id(nb_x, nb_y, rank_x - 1, rank_y + 1);
    mesh_comm->corner_id[CORNER_BOTTOM_RIGHT] = helper_get_rank_id(nb_x, nb_y, rank_x + 1, rank_y + 1);

    // If more than 1 on y, need transmission buffer
    if (nb_y > 1 || nb_x > 1)
    {

        uint32_t local_width = width / nb_x;
        uint32_t local_height = height / nb_y;
        uint32_t max_size;

        if (local_width > local_height)
            max_size = local_width;
        else
            max_size = local_height;

        mesh_comm->buffer = static_cast<double *>(malloc(sizeof(double) * DIRECTIONS * max_size));
    }
    else
    {
        mesh_comm->buffer = NULL;
    }

    int neigh[8] = {mesh_comm->left_id,
                    mesh_comm->right_id,
                    mesh_comm->top_id,
                    mesh_comm->bottom_id,
                    mesh_comm->corner_id[CORNER_TOP_LEFT],
                    mesh_comm->corner_id[CORNER_TOP_RIGHT],
                    mesh_comm->corner_id[CORNER_BOTTOM_LEFT],
                    mesh_comm->corner_id[CORNER_BOTTOM_RIGHT]};

                    
    // --- MPI_Neighbot_Alltoall
    int sources[8];
    int destinations[8];
    int degree = 0;

    for (int i = 0; i < 8; i++)
    {
        if (neigh[i] != -1)
        {
            sources[degree] = neigh[i];
            destinations[degree] = neigh[i];
            degree++;
        }
    }

    if (degree == 0)
    {
        mesh_comm->graph_comm = MPI_COMM_SELF;
        mesh_comm->mpi_degree = 0;
        return;
    }
    else
    {
        int *src_ptr = (degree > 0) ? sources : NULL;
        int *dst_ptr = (degree > 0) ? destinations : NULL;

        MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, degree, src_ptr, MPI_UNWEIGHTED, degree, dst_ptr, MPI_UNWEIGHTED,
                                       MPI_INFO_NULL, 0, &mesh_comm->graph_comm);
    }

    memcpy(mesh_comm->neighbors, neigh, 8 * sizeof(int));

    MPI_Dist_graph_neighbors_count(mesh_comm->graph_comm, &mesh_comm->mpi_degree, &mesh_comm->mpi_degree,
                                   MPI_UNWEIGHTED);

    MPI_Dist_graph_neighbors(mesh_comm->graph_comm, mesh_comm->mpi_degree, mesh_comm->mpi_neighbors, MPI_UNWEIGHTED,
                             mesh_comm->mpi_degree, mesh_comm->mpi_neighbors, MPI_UNWEIGHTED);

    for (int i = 0; i < 8; i++)
    {
        mesh_comm->map[i] = -1;

        if (mesh_comm->neighbors[i] == -1)
            continue;

        for (int j = 0; j < mesh_comm->mpi_degree; j++)
        {
            if (mesh_comm->neighbors[i] == mesh_comm->mpi_neighbors[j])
            {
                mesh_comm->map[i] = j;
                break;
            }
        }
    }

    int horiz = (mesh_comm->height - 2) * DIRECTIONS;
    int vert = (mesh_comm->width - 2) * DIRECTIONS;
    int diag = DIRECTIONS;

    // max 8 voisins
    int max_total = 2 * (horiz + vert) + 4 * diag;

    mesh_comm->buf_size = max_total;

    mesh_comm->sendbuf = (double *)malloc(sizeof(double) * max_total);
    mesh_comm->recvbuf = (double *)malloc(sizeof(double) * max_total);

    // -------

    lbm_comm_print(mesh_comm);
}

void lbm_comm_release(lbm_comm_t *mesh_comm)
{
    mesh_comm->x = 0;
    mesh_comm->y = 0;
    mesh_comm->width = 0;
    mesh_comm->height = 0;
    mesh_comm->right_id = -1;
    mesh_comm->left_id = -1;
    if (mesh_comm->buffer != NULL)
    {
        free(mesh_comm->buffer);
    }
    free(mesh_comm->sendbuf);
    free(mesh_comm->recvbuf);
}

#define TOP_TO_BOT 10
#define BOT_TO_TOP 20
#define LEFT_TO_RIGHT 30
#define RIGHT_TO_LEFT 40
#define DIAG_TOPLEFT 50
#define DIAG_TOPRIGHT 60
#define DIAG_BOTLEFT 70
#define DIAG_BOTRIGHT 80

static inline void pack_direction(Mesh *m, int dir, double *buffer, int &idx, int width, int height)
{
    switch (dir)
    {

    case 0: // LEFT
        for (int k = 0; k < DIRECTIONS; k++)
            for (int y = 1; y < height - 1; y++)
                buffer[idx++] = Mesh_get_value(m, 1, y, k);
        break;

    case 1: // RIGHT
        for (int k = 0; k < DIRECTIONS; k++)
            for (int y = 1; y < height - 1; y++)
                buffer[idx++] = Mesh_get_value(m, width - 2, y, k);
        break;

    case 2: // TOP
        for (int k = 0; k < DIRECTIONS; k++)
            for (int x = 1; x < width - 1; x++)
                buffer[idx++] = Mesh_get_value(m, x, 1, k);
        break;

    case 3: // BOTTOM
        for (int k = 0; k < DIRECTIONS; k++)
            for (int x = 1; x < width - 1; x++)
                buffer[idx++] = Mesh_get_value(m, x, height - 2, k);
        break;

    case 4: // TL
        for (int k = 0; k < DIRECTIONS; k++)
            buffer[idx++] = Mesh_get_value(m, 1, 1, k);
        break;

    case 5: // TR
        for (int k = 0; k < DIRECTIONS; k++)
            buffer[idx++] = Mesh_get_value(m, width - 2, 1, k);
        break;

    case 6: // BL
        for (int k = 0; k < DIRECTIONS; k++)
            buffer[idx++] = Mesh_get_value(m, 1, height - 2, k);
        break;

    case 7: // BR
        for (int k = 0; k < DIRECTIONS; k++)
            buffer[idx++] = Mesh_get_value(m, width - 2, height - 2, k);
        break;
    }
}

static inline void unpack_direction(Mesh *m, int dir, double *buffer, int &idx, int width, int height)
{
    switch (dir)
    {

    case 0: // LEFT ghost
        for (int k = 0; k < DIRECTIONS; k++)
            for (int y = 1; y < height - 1; y++)
                Mesh_get_value(m, 0, y, k) = buffer[idx++];
        break;

    case 1: // RIGHT ghost
        for (int k = 0; k < DIRECTIONS; k++)
            for (int y = 1; y < height - 1; y++)
                Mesh_get_value(m, width - 1, y, k) = buffer[idx++];
        break;

    case 2: // TOP ghost
        for (int k = 0; k < DIRECTIONS; k++)
            for (int x = 1; x < width - 1; x++)
                Mesh_get_value(m, x, 0, k) = buffer[idx++];
        break;

    case 3: // BOTTOM ghost
        for (int k = 0; k < DIRECTIONS; k++)
            for (int x = 1; x < width - 1; x++)
                Mesh_get_value(m, x, height - 1, k) = buffer[idx++];
        break;

    case 4:
        for (int k = 0; k < DIRECTIONS; k++)
            Mesh_get_value(m, 0, 0, k) = buffer[idx++];
        break;

    case 5:
        for (int k = 0; k < DIRECTIONS; k++)
            Mesh_get_value(m, width - 1, 0, k) = buffer[idx++];
        break;

    case 6:
        for (int k = 0; k < DIRECTIONS; k++)
            Mesh_get_value(m, 0, height - 1, k) = buffer[idx++];
        break;

    case 7:
        for (int k = 0; k < DIRECTIONS; k++)
            Mesh_get_value(m, width - 1, height - 1, k) = buffer[idx++];
        break;
    }
}

void lbm_comm_halo_exchange(lbm_comm_t *mesh, Mesh *m, int iteration)
{

    if (mesh->mpi_degree == 0 || mesh->graph_comm == MPI_COMM_SELF) {
        return;
    }

    const int N = mesh->mpi_degree;

    int counts[N];
    int displs[N];

    for (int i = 0; i < N; i++) {
        counts[i] = 0;
        displs[i] = 0;
    }

    int horiz = (mesh->height - 2) * DIRECTIONS;
    int vert = (mesh->width - 2) * DIRECTIONS;
    int diag = DIRECTIONS;


    if (mesh->map[0] != -1)
        counts[mesh->map[0]] = horiz; // left
    if (mesh->map[1] != -1)
        counts[mesh->map[1]] = horiz; // right
    if (mesh->map[2] != -1)
        counts[mesh->map[2]] = vert; // top
    if (mesh->map[3] != -1)
        counts[mesh->map[3]] = vert; // bottom
    if (mesh->map[4] != -1)
        counts[mesh->map[4]] = diag; // TL
    if (mesh->map[5] != -1)
        counts[mesh->map[5]] = diag; // TR
    if (mesh->map[6] != -1)
        counts[mesh->map[6]] = diag; // BL
    if (mesh->map[7] != -1)
        counts[mesh->map[7]] = diag; // BR

    // displacements
    displs[0] = 0;
    for (int i = 1; i < N; i++)
        displs[i] = displs[i - 1] + counts[i - 1];

    int total = displs[N - 1] + counts[N - 1];

    double *sendbuf = mesh->sendbuf;
    double *recvbuf = mesh->recvbuf;

    // =========================================================
    // PACK

    for (int d = 0; d < 8; d++)
    {
        int mpi_idx = mesh->map[d];
        if (mpi_idx == -1)
            continue;

        int idx = displs[mpi_idx];
        pack_direction(m, d, sendbuf, idx, mesh->width, mesh->height);
    }


    MPI_Neighbor_alltoallv(sendbuf, counts, displs, MPI_DOUBLE, recvbuf, counts, displs, MPI_DOUBLE, mesh->graph_comm);

    // =========================================================
    // UNPACK

    for (int d = 0; d < 8; d++)
    {
        int mpi_idx = mesh->map[d];
        if (mpi_idx == -1)
            continue;

        int idx = displs[mpi_idx];
        unpack_direction(m, d, recvbuf, idx, mesh->width, mesh->height);
    }
}

void save_frame_all_domain(FILE *fp, Mesh *source_mesh, Mesh *temp)
{
    int comm_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // If we have more than one process
    if (1 < comm_size)
    {
        if (rank == RANK_MASTER)
        {
            // Rank 0 renders its local Mesh
            save_frame(fp, source_mesh);
            // Rank 0 receives & render other processes meshes
            for (ssize_t i = 1; i < comm_size; i++)
            {
                MPI_Status status;
                MPI_Recv(temp->cells, source_mesh->width * source_mesh->height * DIRECTIONS, MPI_DOUBLE, i, 0,
                         MPI_COMM_WORLD, &status);
                save_frame(fp, temp);
            }
        }
        else
        {
            // All other ranks send their local mesh
            MPI_Send(source_mesh->cells, source_mesh->width * source_mesh->height * DIRECTIONS, MPI_DOUBLE, RANK_MASTER,
                     0, MPI_COMM_WORLD);
        }
    }
    else
    {
        // Only 0 renders its local mesh
        save_frame(fp, source_mesh);
    }
}
