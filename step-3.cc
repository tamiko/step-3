/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */

#include <deal.II/distributed/tria.h> // CHANGES
#include <deal.II/grid/grid_out.h>    // CHANGES

#include <deal.II/lac/generic_linear_algebra.h> // CHANGES
namespace LA = dealii::LinearAlgebraPETSc;      // CHANGES
#include <deal.II/lac/sparsity_tools.h>         // CHANGES
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;



class Step3
{
public:
  Step3();

  void run();


private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  parallel::distributed::Triangulation<2> triangulation;

  FE_Q<2>       fe;
  DoFHandler<2> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  LA::MPI::SparseMatrix system_matrix;
  LA::MPI::Vector       solution;
  LA::MPI::Vector       system_rhs;
};


Step3::Step3()
  : triangulation(MPI_COMM_WORLD)
  , fe(1)
  , dof_handler(triangulation)
{}



void Step3::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(5);

  {
    const auto mpi_rank =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    std::ofstream out("triangulation-" + std::to_string(mpi_rank) + ".inp");
    GridOut       grid_out;
    grid_out.write_ucd(triangulation, out);
  }

  const auto mpi_rank =
    dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  std::cout << "Rank " + std::to_string(mpi_rank) + "  #cells = "
            << triangulation.n_active_cells() << std::endl;
}



void Step3::setup_system()
{
  // distribute dofs:

  dof_handler.distribute_dofs(fe);

  const auto mpi_rank =
    dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  std::cout << "Rank " + std::to_string(mpi_rank) + "  #dofs  = "
            << dof_handler.n_locally_owned_dofs() << std::endl;

  // get locally owned and locally relevant dofs:

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  // initialize vectors:

  solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

  // create constraints:

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           constraints);
  constraints.close();

  // initialize matrix:

  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             MPI_COMM_WORLD,
                                             locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       MPI_COMM_WORLD);
}



void Step3::assemble_system()
{
  QGauss<2>   quadrature_formula(fe.degree + 1);
  FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      /* skip all cells that are not locally owned: */
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            1. *                                // f(x_q)
                            fe_values.JxW(q_index));            // dx
        }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}



void Step3::solve()
{
  SolverControl solver_control(1000, 1e-12);
  LA::SolverCG  solver(solver_control, MPI_COMM_WORLD);

  PETScWrappers::PreconditionNone preconditioner(system_matrix);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);
}



void Step3::output_results() const
{
  LA::MPI::Vector locally_relevant_solution(locally_owned_dofs,
                                            locally_relevant_dofs,
                                            MPI_COMM_WORLD);
  locally_relevant_solution = solution;

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(locally_relevant_solution, "solution");
  data_out.build_patches();

  const auto mpi_rank =
    dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  std::ofstream output("solution-" + std::to_string(mpi_rank) + ".vtk");
  data_out.write_vtk(output);
}



void Step3::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}



int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  deallog.depth_console(2);

  Step3 laplace_problem;
  laplace_problem.run();

  return 0;
}
