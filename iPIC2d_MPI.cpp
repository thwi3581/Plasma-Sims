#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <mpi.h>
#include <algorithm>
#include <array>

// ** LINSPACE FUCNTION **
std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> result;
    if (num == 0) return result;
    if (num == 1) {
        result.push_back(start);
        return result;
    }

    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result.push_back(start + i * step);
    }
    return result;
}

// ** PARTICLES ** 
struct Particle {
    double px;
    double py;
    double vx; 
    double vy; 
    int pm; 
};

// ** POISSON FINITE DIFFERENCE SOLVER **
void Poisson_solver(std::vector<std::vector<double>>& phi, const std::vector<std::vector<double>>& rho, 
    int NGx_domain, int NGy_domain, double dx, double dy, int left, int right, int up, int down, 
    MPI_Comm cart_comm, int max_iteration = 10000, double tol = 1e-6) {
    
    std::vector<std::vector<double>> phi_new = phi;

    std::vector<double> send_up_buffer_phi(NGx_domain);
    std::vector<double> send_down_buffer_phi(NGx_domain);
    std::vector<double> send_left_buffer_phi(NGy_domain);
    std::vector<double> send_right_buffer_phi(NGy_domain); 
    
    std::vector<double> recv_down_buffer_phi(NGx_domain);
    std::vector<double> recv_up_buffer_phi(NGx_domain);
    std::vector<double> recv_right_buffer_phi(NGy_domain);
    std::vector<double> recv_left_buffer_phi(NGy_domain); 

    for (int iter = 0; iter < max_iteration; ++iter) {
        for (int i = 1; i <= NGx_domain; ++i) {
            send_down_buffer_phi[i-1] = phi[i][1]; 
            send_up_buffer_phi[i-1] = phi[i][NGy_domain]; 
        }

        MPI_Sendrecv(send_down_buffer_phi.data(), NGx_domain, MPI_DOUBLE, down, 0, recv_up_buffer_phi.data(), NGx_domain, MPI_DOUBLE, up, 0, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_up_buffer_phi.data(), NGx_domain, MPI_DOUBLE, up, 1, recv_down_buffer_phi.data(), NGx_domain, MPI_DOUBLE, down, 1, cart_comm, MPI_STATUS_IGNORE);

        for (int i = 1; i <= NGx_domain; ++i) {
            phi[i][NGy_domain + 1] = recv_up_buffer_phi[i-1];
            phi[i][0] = recv_down_buffer_phi[i-1];
        }

        for (int j = 1; j <= NGy_domain; ++j) {
            send_left_buffer_phi[j-1] = phi[1][j]; 
            send_right_buffer_phi[j-1] = phi[NGx_domain][j]; 
        }

        MPI_Sendrecv(send_left_buffer_phi.data(), NGy_domain, MPI_DOUBLE, left, 2, recv_right_buffer_phi.data(), NGy_domain, MPI_DOUBLE, right, 2, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_right_buffer_phi.data(), NGy_domain, MPI_DOUBLE, right, 3, recv_left_buffer_phi.data(), NGy_domain, MPI_DOUBLE, left, 3, cart_comm, MPI_STATUS_IGNORE);
        
        for (int j = 1; j <= NGy_domain; ++j) {
            phi[NGx_domain + 1][j] = recv_right_buffer_phi[j - 1]; 
            phi[0][j] = recv_left_buffer_phi[j - 1];
        }

        double local_error = 0; 

        for (int i = 1; i <= NGx_domain; ++i) { 
            for (int j = 1; j <= NGy_domain; ++j) {
                phi_new[i][j] = (0.25) * ((phi[i + 1][j]) + (phi[i - 1][j]) + (phi[i][j + 1]) + (phi[i][j - 1]) + ((dx * dx) * rho[i][j])); 
                double value_change = std::abs(phi_new[i][j] - phi[i][j]);
                if (value_change > local_error) {
                    local_error = value_change;
                }
            }
        }

        double global_error; 
        MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

        if (global_error < tol) {
            break;
        }
        phi = phi_new; 
    }
}

// ** fIELD FINITE DIFFERENCE SOLVER **
void Field_solver(std::vector<std::vector<double>> &phi, std::vector<std::vector<double>> &E_x, std::vector<std::vector<double>> &E_y,
    int NGx_domain, int NGy_domain, double dx, double dy) {

    for (int i = 1; i <= NGx_domain; ++i) { 
        for (int j = 1; j <= NGy_domain; ++j) {
            E_x[i][j] = -(phi[i +1][j] - phi[i -1][j]) / (2 * dx);
            E_y[i][j] = -(phi[i][j +1] - phi[i][j -1]) / (2 * dy); 
        }
    }
}

struct Test_Box { 
    double x_min = 2 * M_PI / 3;
    double x_max = 5 * M_PI / 3; 
    double y_min = 2 * M_PI / 3; 
    double y_max = 5 * M_PI / 3;

    double Q_surface = 0.0; // initial surface charge
    double phi_surface = 0.0; // grounded conducting box
};

// ** MAIN FUNCTION **
int main(int argc, char** argv) {

double L = 2.0 * M_PI; // total domain size 

int NGx = 32; // number of grid points in x 
int NGy = 32; // number of grid points in y
int NGx_domain = NGx / 2; // number of grid points in x per domain
int NGy_domain = NGy / 2; // number of grid points in y per domain

double dx = L / NGx; // step size in x
double dy = L / NGy; // step size in y

int N = 1000; // total # of particles

double dt = 0.5; // time step size
int NT = 500; // # of time steps 
int NTOUT = 25; // # of output times

double WP = 1.0; // plasma frequency
double QM = -1.0; // charge per mass 

double V0 = 0.2; // drift velocity amplitude 
double XP1 = 1.0;

double V1 = 0.0; // velocity perturbation amplitude
int mode = 1;

double Q = (WP * WP) / ((QM * N) / (L * L)); // charge per particle 
double rho_back = (-Q * N) / (L * L); // background charge density

//random number generator 
std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<> normal_dist(0.0, 1.0);


MPI_Init(&argc, &argv); // MPI initialization

int global_rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

int dims[2] = {2, 2};
int periods[2] = {1, 1};
int reorder = 0;
MPI_Comm cart_comm;

MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm); // MPI 2x2 cartesian grid decomposition w/ periodic boundary condidtions

int rank; 
MPI_Comm_rank(cart_comm, &rank);
int coords[2];

MPI_Cart_coords(cart_comm, rank, 2, coords);

int up; 
int down; 
int left; 
int right;
MPI_Cart_shift(cart_comm, 0, 1, &left, &right); 
MPI_Cart_shift(cart_comm, 1, 1, &down, &up);  

double x0, xF, y0, yF; 
double domainLengthx = L / dims[0]; 
double domainLengthy = L / dims[1];

x0 = coords[0] * domainLengthx; 
xF = (coords[0] + 1) * domainLengthx;
y0 = coords[1] * domainLengthy; 
yF = (coords[1] + 1) * domainLengthy;

std::vector <Particle> particles;
int N_domain = N / size; // number of particles in each domain

// **   DISCRETIZED CARTESIAN CHARGE DENSITY GRID (+2 in each direction for ghost cells) ** 
std::vector<std::vector<double>> rho(NGx_domain + 2, std::vector<double>(NGy_domain + 2, 0.0));


// **   DISCRETIZED CARTESIAN POTENTIAL GRID (+2 in each direction for ghost cells) **
std::vector<std::vector<double>> phi(NGx_domain + 2, std::vector<double>(NGy_domain + 2, 0.0));

// ** X AND Y FIELD-COMPONENT GRIDS **
std::vector<std::vector<double>> E_x(NGx_domain + 2, std::vector<double>(NGy_domain + 2, 0.0));
std::vector<std::vector<double>> E_y(NGx_domain + 2, std::vector<double>(NGy_domain + 2, 0.0));
 
int rows = 10;
int M = N_domain / rows;
std::vector<double> px = linspace(x0, xF, M);
std::vector<double> py = linspace(y0, yF, rows);

double VT = 0.0; // thermal velocity parameter (0 for cold plasma)

Test_Box box; 
std::uniform_real_distribution<> uniform_x(x0, xF);
std::uniform_real_distribution<> uniform_y(y0, yF);
int particles_on_box = 0; 

// PARTICLE POSTITION AND VELOCITY INITIALIZATION
for (int i = 0; i < M; ++i) {
    for (int j = 0; j < rows; ++j) {
        Particle p;
        p.px = px[i];
        p.py = py[j];
        p.vx = VT * normal_dist(gen);
        p.vy = VT * normal_dist(gen);

        if (p.px >= box.x_min && p.px <= box.x_max && p.py >= box.y_min && p.py <= box.y_max) {
            do  {
                p.px = uniform_x(gen);
                p.py = uniform_y(gen); 
            } while (p.px >= box.x_min && p.px <= box.x_max && p.py >= box.y_min && p.py <= box.y_max); 
        }
        particles.push_back(p);
    }
}

// PARTICLE POSITION AND VELOCITY PERTURBATIONS
for (size_t i = 0; i < particles.size(); ++i) {
    particles[i].pm = 1 - 2 * (i % 2);
    particles[i].vx = particles[i].vx + (V0 * particles[i].pm);
    particles[i].vx = particles[i].vx + (V1 * sin(2 * M_PI * (particles[i].px / (L * mode))));
    particles[i].px = particles[i].px + (XP1 * (L/N) * sin(2 * M_PI * (particles[i].px / (L * mode))));
}

for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
    double px_local = particles[i].px - x0;
    double py_local = particles[i].py - y0;

    double px_grid = px_local / dx;
    double py_grid = py_local / dy;

    // finding closest x and y indices in left and down directions
    int i1 = static_cast<int>(std::floor(px_grid - 0.5));
    int j1 = static_cast<int>(std::floor(py_grid - 0.5));

    //clamping negative indices 
    i1 = std::max(0, std::min(i1, NGx_domain - 2));
    j1 = std::max(0, std::min(j1, NGy_domain - 2));

    std::array<int, 2> ix = {i1, i1 + 1};
    std::array<int, 2> jy = {j1, j1 + 1};

    double frazi1 = px_grid - i1 - 0.5;
    double frazj1 = py_grid - j1 - 0.5;

    // weight clamping to ensure charge conservation
    frazi1 = std::min(std::max(frazi1, 0.0), 1.0);
    frazj1 = std::min(std::max(frazj1, 0.0), 1.0);

    // calcualtion of grid cell weights
    double w_ileft = 1 - frazi1;
    double w_iright = frazi1; 
    double w_jdown = 1 - frazj1; 
    double w_jup = frazj1;

    std::array<double, 2> frazx = {w_ileft, w_iright};
    std::array<double, 2> frazy = {w_jdown, w_jup};

    // charge depotition loop onto each grid point 
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int indexi = ix[i] + 1; 
            int indexj = jy[j] + 1; 
            if (indexi <= NGx_domain && indexi >= 1 && indexj <= NGy_domain && indexj >=1) {
                rho[indexi][indexj] = rho[indexi][indexj] + ((Q / (dx * dy)) * frazx[i] * frazy[j]);
            }
        }
    }
}

// Adding background charge density to each grid point
for (int i = 1; i <= NGx_domain; ++i) {
    for (int j = 1; j <= NGy_domain; ++j) {
        rho[i][j] = rho[i][j] + (rho_back); 
    }
}

// Ghost cell communication
std::vector<double> send_buffer_left_grid_initial(NGy_domain); 
std::vector<double> send_buffer_right_grid_initial(NGy_domain); 
std::vector<double> send_buffer_up_grid_initial(NGx_domain); 
std::vector<double> send_buffer_down_grid_initial(NGx_domain); 

for (int j = 1; j <= NGy_domain; ++j) { 
    send_buffer_left_grid_initial[j - 1] = rho[1][j];
}

for (int j = 1; j <= NGy_domain; ++j) {
    send_buffer_right_grid_initial[j - 1] = rho[NGx_domain][j];
}

for (int i = 1; i <= NGx_domain; ++i) {
    send_buffer_up_grid_initial[i - 1] = rho[i][NGy_domain];
}

for (int i = 1; i <= NGx_domain; ++i) {
    send_buffer_down_grid_initial[i - 1] = rho[i][1]; 
}

std::vector<double> recv_buffer_left_grid_initial(NGy_domain); 
std::vector<double> recv_buffer_right_grid_initial(NGy_domain);
std::vector<double> recv_buffer_up_grid_initial(NGx_domain); 
std::vector<double> recv_buffer_down_grid_initial(NGx_domain);

MPI_Sendrecv(send_buffer_left_grid_initial.data(), NGy_domain, MPI_DOUBLE, left, 10, recv_buffer_right_grid_initial.data(), NGy_domain, MPI_DOUBLE, right, 10, cart_comm, MPI_STATUS_IGNORE);
MPI_Sendrecv(send_buffer_right_grid_initial.data(), NGy_domain, MPI_DOUBLE, right, 11, recv_buffer_left_grid_initial.data(), NGy_domain, MPI_DOUBLE, left, 11, cart_comm, MPI_STATUS_IGNORE); 
MPI_Sendrecv(send_buffer_up_grid_initial.data(), NGx_domain, MPI_DOUBLE, up, 12, recv_buffer_down_grid_initial.data(), NGx_domain, MPI_DOUBLE, down, 12, cart_comm, MPI_STATUS_IGNORE); 
MPI_Sendrecv(send_buffer_down_grid_initial.data(), NGx_domain, MPI_DOUBLE, down, 13, recv_buffer_up_grid_initial.data(), NGx_domain, MPI_DOUBLE, up, 13, cart_comm, MPI_STATUS_IGNORE);
    
for (int j = 1; j <= NGy_domain; ++j) {
    rho[0][j] = recv_buffer_left_grid_initial[j - 1];
    rho[NGx_domain + 1][j] = recv_buffer_right_grid_initial[j - 1];
}

for (int i = 1; i <= NGx_domain; ++i) {
    rho[i][0] = recv_buffer_down_grid_initial[i - 1];
    rho[i][NGy_domain + 1] = recv_buffer_up_grid_initial[i - 1];
}

// ** INITIAL JACOBI ITERATIVE FIELD SOLVER USING rho[i][j] AND phi[i][j] **
Poisson_solver(phi, rho, NGx_domain, NGy_domain, dx, dy, left, right, up, down, cart_comm);

// ** INITIAL FINITE DIFFERENCE FIELD SOLVER CALL ** 
Field_solver(phi, E_x, E_y, NGx_domain, NGy_domain, dx, dy);

// ** INITIAl GRID TO PARTICLE INTERPOLATION **
for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
    double px_local = particles[i].px - x0;
    double py_local = particles[i].py - y0;

    double px_grid = px_local / dx;
    double py_grid = py_local / dy;

    int i1 = static_cast<int>(std::floor(px_grid - 0.5));
    int j1 = static_cast<int>(std::floor(py_grid - 0.5));

    std::array<int, 2> ix = {i1, i1 + 1};
    std::array<int, 2> jy = {j1, j1 + 1};

    double frazi1 = px_grid - i1 - 0.5;
    double frazj1 = py_grid - j1 - 0.5;

    double w_ileft = 1 - frazi1;
    double w_iright = frazi1; 
    double w_jdown = 1 - frazj1; 
    double w_jup = frazj1; 

    std::array<double, 2> frazx = {w_ileft, w_iright};
    std::array<double, 2> frazy = {w_jdown, w_jup};

    double Ex_particle = 0.0;
    double Ey_particle = 0.0;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int indexi = ix[i] + 1;
            int indexj = jy[j] + 1;

            if (indexi >= 1 && indexi <= NGx_domain && indexj >= 1 && indexj <= NGy_domain) {
                Ex_particle += E_x[indexi][indexj] * frazx[i] * frazy[j];
                Ey_particle += E_y[indexi][indexj] * frazx[i] * frazy[j];
            }
        }
    }

    particles[i].vx = particles[i].vx + (QM * Ex_particle * dt / 2); // Half Step Initialization for x-velocity
    particles[i].vy = particles[i].vy + (QM * Ey_particle * dt / 2); // Hald Step INitialization for y-velocity
}

// ** MAIN CYCLE **
for (int t = 0; t < NT; ++t) {

    // resetting grids for each iteration
    for (int i = 0; i < NGx_domain + 2; ++i) { 
        for (int j = 0; j < NGy_domain + 2; ++j) { 
            rho[i][j] = 0.0;
            phi[i][j] = 0.0;
            E_x[i][j] = 0.0;
            E_y[i][j] = 0.0;
        }
    }

    // ** PARTICLE MOVER **
    std::vector<int> indices_to_left; // Collecting indices of particles moving out of the left boundary of the domain
    std::vector<int> indices_to_right; // Collecting indices of particles moving out of the right boundary of the domain

    for (int i = static_cast<int>(particles.size()) - 1; i >= 0; --i) {
        particles[i].px = particles[i].px + (particles[i].vx * dt);

        if (particles[i].px < 0) particles[i].px += L;
        else if (particles[i].px >= L) particles[i].px -= L;

        if (particles[i].px < x0) { 
            indices_to_left.push_back(i);
        }
        else if (particles[i].px >= xF) {
            indices_to_right.push_back(i);
        }
    }

    std::vector<double> send_buffer_left; // creating buffers with particle data for MPI communication
    std::vector<double> send_buffer_right;

    for (int idx : indices_to_left) {
        const Particle& p = particles[idx];
        send_buffer_left.push_back(p.px);
        send_buffer_left.push_back(p.py);
        send_buffer_left.push_back(p.vx);
        send_buffer_left.push_back(p.vy);
        send_buffer_left.push_back(static_cast<double>(p.pm));
    }

    for (int idx : indices_to_right) {
        const Particle& p = particles[idx];
        send_buffer_right.push_back(p.px);
        send_buffer_right.push_back(p.py);
        send_buffer_right.push_back(p.vx);
        send_buffer_right.push_back(p.vy);
        send_buffer_right.push_back(static_cast<double>(p.pm));
    }

    // erasing particles that will ve sent to other MPI domains
    std::vector<int> indices_to_erase = indices_to_left;
    indices_to_erase.insert(indices_to_erase.end(), indices_to_right.begin(), indices_to_right.end());

    std::sort(indices_to_erase.begin(), indices_to_erase.end());
    indices_to_erase.erase(std::unique(indices_to_erase.begin(), indices_to_erase.end()), indices_to_erase.end());

    for (int i = static_cast<int>(indices_to_erase.size()) - 1; i >= 0; --i) {
        int idx = indices_to_erase[i];
        if (idx >= 0 && idx < static_cast<int>(particles.size())) {
            particles.erase(particles.begin() + idx);
            }
    }

    int send_left_count = indices_to_left.size();
    int send_right_count = indices_to_right.size(); 

    int recv_left_count, recv_right_count;

    MPI_Sendrecv(&send_right_count, 1, MPI_INT, right, 0, &recv_left_count, 1, MPI_INT, left, 0, cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&send_left_count, 1, MPI_INT, left, 1, &recv_right_count, 1, MPI_INT, right, 1, cart_comm, MPI_STATUS_IGNORE);
    
    std::vector<double> recv_buffer_left(5 * recv_left_count);
    std::vector<double> recv_buffer_right(5 * recv_right_count);
    
    MPI_Sendrecv(send_buffer_right.data(), 5 * send_right_count, MPI_DOUBLE, right, 2, recv_buffer_left.data(), 5 * recv_left_count, MPI_DOUBLE, left, 2, cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_buffer_left.data(), 5 * send_left_count, MPI_DOUBLE, left, 3, recv_buffer_right.data(), 5 * recv_right_count, MPI_DOUBLE, right, 3, cart_comm, MPI_STATUS_IGNORE);

    for (size_t i = 0; i < recv_buffer_left.size(); i += 5) {
        Particle p; 
        p.px = recv_buffer_left[i];
        p.py = recv_buffer_left[i + 1];
        p.vx = recv_buffer_left[i + 2];
        p.vy = recv_buffer_left[i + 3];
        p.pm = static_cast<int>(recv_buffer_left[i + 4]);
        particles.push_back(p);
    }

    for (size_t i = 0; i < recv_buffer_right.size(); i += 5) {
        Particle p; 
        p.px = recv_buffer_right[i];
        p.py = recv_buffer_right[i + 1];
        p.vx = recv_buffer_right[i + 2];
        p.vy = recv_buffer_right[i + 3];
        p.pm = static_cast<int>(recv_buffer_right[i + 4]);
        particles.push_back(p); 

    }

    std::vector<int> indices_up; 
    std::vector<int> indices_down;

    for (int i = static_cast<int>(particles.size()) - 1; i >= 0; --i) {
        particles[i].py = particles[i].py + (particles[i].vy * dt);

        if (particles[i].py < y0) {
            particles[i].py = particles[i].py + domainLengthy; 
            indices_down.push_back(i); 
        }
        else if (particles[i].py >= yF) {
            particles[i].py = particles[i].py - domainLengthy;
            indices_up.push_back(i);
        }
    }

    std::vector<double> send_buffer_up; 
    std::vector<double> send_buffer_down; 

    for (int idx : indices_up) {
        const Particle& p = particles[idx];
        send_buffer_up.push_back(p.px);
        send_buffer_up.push_back(p.py); 
        send_buffer_up.push_back(p.vx); 
        send_buffer_up.push_back(p.vy); 
        send_buffer_up.push_back(static_cast<double>(p.pm));
    }

    for (int idx : indices_down) {
        const Particle& p = particles[idx];
        send_buffer_down.push_back(p.px);
        send_buffer_down.push_back(p.py);
        send_buffer_down.push_back(p.vx);
        send_buffer_down.push_back(p.vy);
        send_buffer_down.push_back(static_cast<double>(p.pm));
    }

    indices_to_erase.clear(); 
    indices_to_erase = indices_up;
    indices_to_erase.insert(indices_to_erase.end(), indices_down.begin(), indices_down.end());

    std::sort(indices_to_erase.begin(), indices_to_erase.end());
    indices_to_erase.erase(std::unique(indices_to_erase.begin(), indices_to_erase.end()), indices_to_erase.end());

    for (int i = static_cast<int>(indices_to_erase.size()) - 1; i >= 0; --i) {
        int idx = indices_to_erase[i];
        if (idx >= 0 && idx < static_cast<int>(particles.size())) {
            particles.erase(particles.begin() + idx);
            }
    }

    int send_up_count = indices_up.size(); 
    int send_down_count = indices_down.size(); 

    int recv_up_count, recv_down_count; 

    MPI_Sendrecv(&send_up_count, 1, MPI_INT, up, 0, &recv_down_count, 1, MPI_INT, down, 0, cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&send_down_count, 1, MPI_INT, down, 1, &recv_up_count, 1, MPI_INT, up, 1, cart_comm, MPI_STATUS_IGNORE);

    std::vector<double> recv_buffer_down(5 * recv_down_count);
    std::vector<double> recv_buffer_up(5 * recv_up_count);

    MPI_Sendrecv(send_buffer_up.data(), 5 * send_up_count, MPI_DOUBLE, up, 2, recv_buffer_down.data(), 5 * recv_down_count, MPI_DOUBLE, down, 2, cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_buffer_down.data(), 5 * send_down_count, MPI_DOUBLE, down, 3, recv_buffer_up.data(), 5 * recv_up_count, MPI_DOUBLE, up, 3, cart_comm, MPI_STATUS_IGNORE);

    for (size_t i = 0; i < recv_buffer_down.size(); i += 5) {
        Particle p; 
        p.px = recv_buffer_down[i];
        p.py = recv_buffer_down[i + 1];
        p.vx = recv_buffer_down[i + 2];
        p.vy = recv_buffer_down[i + 3];
        p.pm = static_cast<int>(recv_buffer_down[i + 4]);
        particles.push_back(p);
    }

    for (size_t i = 0; i < recv_buffer_up.size(); i += 5) {
        Particle p; 
        p.px = recv_buffer_up[i];
        p.py = recv_buffer_up[i + 1];
        p.vx = recv_buffer_up[i + 2];
        p.vy = recv_buffer_up[i + 3];
        p.pm = static_cast<int>(recv_buffer_up[i + 4]);
        particles.push_back(p); 
    }

    for (int i = static_cast<int>(particles.size()) - 1; i >= 0; --i) {
        if (particles[i].px >= box.x_min && particles[i].px <= box.x_max && particles[i].py >= box.y_min && particles[i].py <= box.y_max) { 
            box.Q_surface += Q; 
            particles.erase(particles.begin() + i);
            particles_on_box += 1;
        }
    }

    // ** PARTICLE TO GRID DEPOSTION **
    for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
        double px_local = particles[i].px - x0;
        double py_local = particles[i].py - y0;

        double px_grid = px_local / dx;
        double py_grid = py_local / dy;

        // finding closest x and y indices in left and down directions
        int i1 = static_cast<int>(std::floor(px_grid - 0.5));
        int j1 = static_cast<int>(std::floor(py_grid - 0.5));

        //clamping negative indices 
        i1 = std::max(0, std::min(i1, NGx_domain - 2));
        j1 = std::max(0, std::min(j1, NGy_domain - 2));

        std::array<int, 2> ix = {i1, i1 + 1};
        std::array<int, 2> jy = {j1, j1 + 1};

        double frazi1 = px_grid - i1 - 0.5;
        double frazj1 = py_grid - j1 - 0.5;

        // weight clamping to ensure charge conservation
        frazi1 = std::min(std::max(frazi1, 0.0), 1.0);
        frazj1 = std::min(std::max(frazj1, 0.0), 1.0);

        // calcualtion of grid cell weights
        double w_ileft = 1 - frazi1;
        double w_iright = frazi1; 
        double w_jdown = 1 - frazj1; 
        double w_jup = frazj1; 

        std::array<double, 2> frazx = {w_ileft, w_iright};
        std::array<double, 2> frazy = {w_jdown, w_jup};

    
        // charge depotition loop onto each grid point 
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int indexi = ix[i] + 1; 
                int indexj = jy[j] + 1; 
                if (indexi <= NGx_domain && indexi >= 1 && indexj <= NGy_domain && indexj >=1) {
                    rho[indexi][indexj] = rho[indexi][indexj] + ((Q / (dx * dy)) * frazx[i] * frazy[j]);
                }

                
            }
        }
    }

    // Adding background charge density to each grid point
    for (int i = 1; i <= NGx_domain; ++i) {
        for (int j = 1; j <= NGy_domain; ++j) {
            rho[i][j] = rho[i][j] + (rho_back); 
        }
    }

    // Ghost cell communication
    std::vector<double> send_buffer_left_grid(NGy_domain); 
    std::vector<double> send_buffer_right_grid(NGy_domain); 
    std::vector<double> send_buffer_up_grid(NGx_domain); 
    std::vector<double> send_buffer_down_grid(NGx_domain); 

    for (int j = 1; j <= NGy_domain; ++j) { 
        send_buffer_left_grid[j - 1] = rho[1][j];
    }

    for (int j = 1; j <= NGy_domain; ++j) {
        send_buffer_right_grid[j - 1] = rho[NGx_domain][j];
    }

    for (int i = 1; i <= NGx_domain; ++i) {
        send_buffer_up_grid[i - 1] = rho[i][NGy_domain];
    }

    for (int i = 1; i <= NGx_domain; ++i) {
        send_buffer_down_grid[i - 1] = rho[i][1]; 
    }

    std::vector<double> recv_buffer_left_grid(NGy_domain); 
    std::vector<double> recv_buffer_right_grid(NGy_domain);
    std::vector<double> recv_buffer_up_grid(NGx_domain); 
    std::vector<double> recv_buffer_down_grid(NGx_domain);

    MPI_Sendrecv(send_buffer_left_grid.data(), NGy_domain, MPI_DOUBLE, left, 10, recv_buffer_right_grid.data(), NGy_domain, MPI_DOUBLE, right, 10, cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_buffer_right_grid.data(), NGy_domain, MPI_DOUBLE, right, 11, recv_buffer_left_grid.data(), NGy_domain, MPI_DOUBLE, left, 11, cart_comm, MPI_STATUS_IGNORE); 
    MPI_Sendrecv(send_buffer_up_grid.data(), NGx_domain, MPI_DOUBLE, up, 12, recv_buffer_down_grid.data(), NGx_domain, MPI_DOUBLE, down, 12, cart_comm, MPI_STATUS_IGNORE); 
    MPI_Sendrecv(send_buffer_down_grid.data(), NGx_domain, MPI_DOUBLE, down, 13, recv_buffer_up_grid.data(), NGx_domain, MPI_DOUBLE, up, 13, cart_comm, MPI_STATUS_IGNORE);
    
    for (int j = 1; j <= NGy_domain; ++j) {
        rho[0][j] = recv_buffer_left_grid[j - 1];
        rho[NGx_domain + 1][j] = recv_buffer_right_grid[j - 1];
    }

    for (int i = 1; i <= NGx_domain; ++i) {
        rho[i][0] = recv_buffer_down_grid[i - 1];
        rho[i][NGy_domain + 1] = recv_buffer_up_grid[i - 1];
    }

    // ** JACOBI ITERATIVE FIELD SOLVER USING rho[i][j] AND phi[i][j] **
    Poisson_solver(phi, rho, NGx_domain, NGy_domain, dx, dy, left, right, up, down, cart_comm);

    // ** FINITE DIFFERENCE FIELD SOLVER CALL ** 
    Field_solver(phi, E_x, E_y, NGx_domain, NGy_domain, dx, dy);

    // ** GRID TO PARTICLE INTERPOLATION **
    for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
        double px_local = particles[i].px - x0;
        double py_local = particles[i].py - y0;

        double px_grid = px_local / dx;
        double py_grid = py_local / dy;

        int i1 = static_cast<int>(std::floor(px_grid - 0.5));
        int j1 = static_cast<int>(std::floor(py_grid - 0.5));

        std::array<int, 2> ix = {i1, i1 + 1};
        std::array<int, 2> jy = {j1, j1 + 1};

        double frazi1 = px_grid - i1 - 0.5;
        double frazj1 = py_grid - j1 - 0.5;

        double w_ileft = 1 - frazi1;
        double w_iright = frazi1; 
        double w_jdown = 1 - frazj1; 
        double w_jup = frazj1;

        double Ex_particle = 0.0;
        double Ey_particle = 0.0;

        std::array<double, 2> frazx = {w_ileft, w_iright};
        std::array<double, 2> frazy = {w_jdown, w_jup};

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int indexi = ix[i] + 1;
                int indexj = jy[j] + 1;

                if (indexi >= 1 && indexi <= NGx_domain && indexj >= 1 && indexj <= NGy_domain) {
                    Ex_particle += E_x[indexi][indexj] * frazx[i] * frazy[j];
                    Ey_particle += E_y[indexi][indexj] * frazx[i] * frazy[j];
                }
            }
        }

        particles[i].vx = particles[i].vx + (QM * Ex_particle * dt);
        particles[i].vy = particles[i].vy + (QM * Ey_particle * dt);
    }
}
MPI_Barrier(MPI_COMM_WORLD);

int free_particles = static_cast<int>(particles.size());
int total_free_particles = 0; 
MPI_Reduce(&free_particles, &total_free_particles, 1, MPI_INT, MPI_SUM, 0, cart_comm);

int total_particles_on_box = 0; 
MPI_Reduce(&particles_on_box, &total_particles_on_box, 1, MPI_INT, MPI_SUM, 0, cart_comm);

double total_box_charge = 0; 
MPI_Reduce(&box.Q_surface, &total_box_charge, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm); 

if (rank == 0) std::cout << "Total Charge on Box Surface = " << total_box_charge << std::endl;
if (rank == 0) std::cout << "The Total Number of Remaining Particles is: " << total_free_particles + total_particles_on_box << std::endl;

MPI_Finalize(); 
return 0;
}

// mpic++ -O2 -std=c++17 iPIC2d_MPI.cpp -o iPIC2d_MPI -- TERMINAL BUILD COMMAND