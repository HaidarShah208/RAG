import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, UpdateDateColumn } from 'typeorm';

@Entity('all_files')
export class AllFiles {
  @PrimaryGeneratedColumn('uuid')
    id!: string;

  @Column({ type: 'varchar', length: 255 })
    filename!: string;

  @Column({ type: 'varchar', length: 100 })
    originalName!: string;

  @Column({ type: 'varchar', length: 50 })
    mimeType!: string;

  @Column({ type: 'int' })
    size!: number;

  @Column({ type: 'text' })
    filePath!: string;

  @Column({ type: 'varchar', length: 255, nullable: true })
    description!: string;

  @Column({ type: 'varchar', length: 100, default: 'pdf' })
    fileType!: string;

  @CreateDateColumn()
    createdAt!: Date;

  @UpdateDateColumn()
    updatedAt!: Date;
} 